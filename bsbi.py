import os
import pickle
import contextlib
import heapq
import time
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs, compute_max_impact_tfidf, compute_max_impact_bm25, compute_single_term_score
from compression import EliasGammaPostings, StandardPostings, VBEPostings
from tqdm import tqdm


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index", wand_config={"use_wand": False, "scoring_function": "bm25"}):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.use_wand = wand_config.get("use_wand", False)
        self.scoring_function = wand_config.get("scoring_function", "bm25")

        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                for token in f.read().split():
                    td_pairs.append(
                        (self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        N = len(set(doc_id for _, doc_id in td_pairs))
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]

            df = len(sorted_doc_id)
            if self.use_wand:
                if self.scoring_function == "tfidf":
                    max_impact = compute_max_impact_tfidf(
                        assoc_tf, df, N)
                else:
                    max_impact = compute_max_impact_bm25(
                        assoc_tf, df, N)
                index.append(term_id, sorted_doc_id, assoc_tf, max_impact)
            else:
                index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)),
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * \
                                (1 + math.log(tf))

            # Top-K
            docs = [(score, self.doc_id_map[doc_id])
                    for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.5, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Score = untuk setiap term di query, akumulasikan IDF * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl))))

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)
            4. informasi dl bisa didapat dari doc_length pada merged index, doc_length[doc_id]
            5. informasi avgdl bisa didapat dari doc_length pada merged index, sum(doc_length.values()) / len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            avgdl = sum(merged_index.doc_length.values()) / \
                len(merged_index.doc_length)
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        dl = merged_index.doc_length[doc_id]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            idf = math.log(N / df)
                            score = idf * \
                                ((tf * (k1 + 1)) /
                                 (tf + k1 * (1 - b + b * (dl / avgdl))))
                            scores[doc_id] += score

            # Top-K
            docs = [(score, self.doc_id_map[doc_id])
                    for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_wand_optimized(self, query, k=10, scoring='tfidf', k1=1.5, b=0.75):
        """
        Optimized WAND with pivot selection and early termination.
        Follows the classic WAND algorithm structure.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        query_terms = [self.term_id_map[word]
                       for word in query.split() if word in self.term_id_map.str_to_id]
        if not query_terms:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0:
                return []
            avgdl = sum(merged_index.doc_length.values()) / \
                N if scoring == 'bm25' else 0

            # Pre-load term data
            term_data = []
            for term_id in query_terms:
                if term_id not in merged_index.postings_dict:
                    continue
                meta = merged_index.postings_dict[term_id]
                df = meta[1]
                postings, tf_list = merged_index.get_postings_list(term_id)
                if len(meta) >= 5:
                    upper_bound = meta[4]
                else:
                    # Compute bound using the SAME scoring function
                    if scoring == 'tfidf':
                        upper_bound = compute_max_impact_tfidf(tf_list, df, N)
                    else:  # bm25
                        upper_bound = compute_max_impact_bm25(
                            tf_list, df, N, k1=k1, b=b)

                # Pre-compute IDF
                idf = math.log(N / df) if df > 0 else 0.0

                term_data.append({
                    'term_id': term_id,
                    'postings': postings,
                    'tf_list': tf_list,
                    'upper_bound': upper_bound,
                    'idf': idf,
                    'pointer': 0
                })

            if not term_data:
                return []

            # Sort by upper bound descending
            term_data.sort(key=lambda x: x['upper_bound'], reverse=True)

            # Min-heap for top-K: (score, doc_id)
            top_k = []
            threshold = 0.0

            # Pre-compute doc_length lookup
            doc_len = merged_index.doc_length

            # WAND main loop with pivot
            while True:
                # Find the term with smallest current docID
                active = [(td['postings'][td['pointer']], td)
                          for td in term_data if td['pointer'] < len(td['postings'])]
                if not active:
                    break

                # Sort active terms by current docID
                active.sort(key=lambda x: x[0])
                current_doc = active[0][0]

                # PIVOT SELECTION: find minimal prefix of terms whose bounds sum >= threshold
                cumulative = 0.0
                pivot_idx = -1
                for i, (doc_id, td) in enumerate(active):
                    if doc_id == current_doc:
                        cumulative += td['upper_bound']
                        if cumulative >= threshold:
                            pivot_idx = i
                            break

                if pivot_idx == -1:
                    # Cannot reach threshold: skip to next smallest docID
                    next_doc = active[-1][0]  # largest current docID
                    for td in term_data:
                        while td['pointer'] < len(td['postings']) and td['postings'][td['pointer']] <= next_doc:
                            td['pointer'] += 1
                    continue

                # Evaluate candidate document (current_doc)
                score = 0.0
                dl = doc_len.get(current_doc, 0)
                for doc_id, td in active:
                    if doc_id == current_doc:
                        tf = td['tf_list'][td['pointer']]
                        score += compute_single_term_score(
                            tf, td['idf'], dl, avgdl, scoring, k1, b)

                # Update top-K
                if len(top_k) < k:
                    heapq.heappush(top_k, (score, current_doc))
                    if len(top_k) == k:
                        threshold = top_k[0][0]
                elif score > threshold:
                    heapq.heapreplace(top_k, (score, current_doc))
                    threshold = top_k[0][0]

                # Advance all pointers at current_doc
                for td in term_data:
                    if td['pointer'] < len(td['postings']) and td['postings'][td['pointer']] == current_doc:
                        td['pointer'] += 1

            # Return formatted results
            results = [(score, self.doc_id_map[doc_id])
                       for score, doc_id in sorted(top_k, reverse=True)]
            return results

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=EliasGammaPostings,
                              output_dir='index', wand_config={"use_wand": True, "scoring_function": "bm25"})
    BSBI_instance.index()  # memulai indexing!
