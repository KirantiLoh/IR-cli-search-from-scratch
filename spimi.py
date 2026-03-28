import os
import pickle
import contextlib
import heapq
import time
import math
import sys

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs, compute_max_impact_tfidf, compute_max_impact_bm25, compute_single_term_score
from compression import EliasGammaPostings, StandardPostings, VBEPostings
from tqdm import tqdm


class SPIMIIndex:
    """
    SPIMI (Single-Pass In-Memory Indexing) Implementation

    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen ke docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Encoding untuk postings list
    index_name(str): Nama dari file yang berisi inverted index
    memory_limit(int): Batas memory untuk setiap block (dalam bytes)
    """

    def __init__(self, data_dir, output_dir, postings_encoding,
                 index_name="main_index", wand_config={"use_wand": False, "scoring_function": "bm25"}, memory_limit_mb=500):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.use_wand = wand_config.get("use_wand", False)
        self.scoring_function = wand_config.get("scoring_function", "bm25")

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""
        os.makedirs(self.output_dir, exist_ok=True)

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

    def _get_memory_usage(self, td_pairs):
        """Estimasi memory usage dari td_pairs"""
        return sys.getsizeof(td_pairs) + sum(sys.getsizeof(pair) for pair in td_pairs)

    def parse_block(self, block_dir_relative):
        """
        SPIMI: Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        PRINSIP SPIMI:
        - Single pass melalui dokumen
        - Tidak perlu sort terms selama parsing
        - Gunakan hash map untuk group postings by term immediately

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
        """
        dir_path = os.path.join(self.data_dir, block_dir_relative)
        td_pairs = []

        # Dapatkan semua file dalam block
        try:
            _, _, filenames = next(os.walk(dir_path))
        except StopIteration:
            return td_pairs

        for filename in sorted(filenames):  # Sort untuk konsistensi
            docname = os.path.join(dir_path, filename)

            # Dapatkan docID (persist across blocks)
            doc_id = self.doc_id_map[docname]

            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                content = f.read()

                # Tokenization sederhana (bisa diganti dengan NLTK/spacy)
                tokens = content.split()

                for token in tokens:
                    token = token.lower().strip()
                    if token:  # Skip empty tokens
                        # Dapatkan termID (persist across blocks)
                        term_id = self.term_id_map[token]
                        td_pairs.append((term_id, doc_id))

            # SPIMI: Check memory usage setelah setiap dokumen
            if self._get_memory_usage(td_pairs) > self.memory_limit:
                # Memory penuh, dump intermediate index dan reset
                # (Untuk implementasi penuh, perlu dump di level class)
                pass

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        SPIMI: Melakukan inversion td_pairs menggunakan HASH MAP.

        PERBEDAAN SPIMI vs BSBI:
        - BSBI: Sort semua (term, doc) pairs dulu, baru build index
        - SPIMI: Gunakan hash map untuk group by term immediately, 
                 tidak perlu sort raw pairs

        CATATAN: Kita masih sort term_keys saat write ke disk agar merge 
        bisa menggunakan heapq.merge, tapi ini bukan external sort seperti BSBI.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk untuk suatu block
        """
        # SPIMI: Gunakan hash map untuk group postings by term immediately
        # Tidak perlu sort td_pairs sebelum proses ini!
        term_postings = {}  # term_id -> {doc_id: tf}

        for term_id, doc_id in td_pairs:
            if term_id not in term_postings:
                term_postings[term_id] = {}
            if doc_id not in term_postings[term_id]:
                term_postings[term_id][doc_id] = 0
            term_postings[term_id][doc_id] += 1

        # Sort term keys hanya saat write ke disk (bukan sort raw pairs)
        # Ini diperlukan agar merge step bisa menggunakan heapq.merge
        for term_id in sorted(term_postings.keys()):
            doc_tf_dict = term_postings[term_id]
            sorted_doc_ids = sorted(doc_tf_dict.keys())
            tf_list = [doc_tf_dict[doc_id] for doc_id in sorted_doc_ids]

            index.append(term_id, sorted_doc_ids, tf_list)

        # Clear memory
        term_postings.clear()

    def merge(self, indices, merged_index):
        """
        SPIMI: Lakukan merging ke semua intermediate inverted indices.

        Ini adalah EXTERNAL MERGE yang efisien karena setiap block index
        sudah sorted by term_id.

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            List of intermediate InvertedIndexReader objects
        merged_index: InvertedIndexWriter
            Final merged InvertedIndexWriter object
        """
        if not indices:
            return

        # heapq.merge expects sorted inputs (which our block indices are)
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])

        try:
            curr_term, postings, tf_list = next(merged_iter)
        except StopIteration:
            return

        for term, postings_, tf_list_ in merged_iter:
            if term == curr_term:
                # Merge postings lists untuk term yang sama dari different blocks
                zip_p_tf = sorted_merge_posts_and_tfs(
                    list(zip(postings, tf_list)),
                    list(zip(postings_, tf_list_))
                )
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                # Write completed term to merged index
                merged_index.append(curr_term, postings, tf_list)
                curr_term, postings, tf_list = term, postings_, tf_list_

        # Write last term
        merged_index.append(curr_term, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Tokenize query dan map ke termIDs
        query_terms = []
        for word in query.split():
            word = word.lower().strip()
            if word in self.term_id_map:
                query_terms.append(self.term_id_map[word])

        if not query_terms:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:

            scores = {}
            N = len(merged_index.doc_length)

            for term in query_terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    postings, tf_list = merged_index.get_postings_list(term)

                    for i in range(len(postings)):
                        doc_id = postings[i]
                        tf = tf_list[i]

                        if doc_id not in scores:
                            scores[doc_id] = 0

                        if tf > 0:
                            # TF-IDF scoring
                            tf_weight = 1 + math.log(tf)
                            idf_weight = math.log(N / df)
                            scores[doc_id] += tf_weight * idf_weight

            # Top-K retrieval
            docs = [(score, self.doc_id_map[doc_id])
                    for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.5, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema BM25.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Tokenize query dan map ke termIDs
        query_terms = []
        for word in query.split():
            word = word.lower().strip()
            if word in self.term_id_map:
                query_terms.append(self.term_id_map[word])

        if not query_terms:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:

            scores = {}
            N = len(merged_index.doc_length)
            avgdl = sum(merged_index.doc_length.values()) / N

            for term in query_terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    postings, tf_list = merged_index.get_postings_list(term)

                    for i in range(len(postings)):
                        doc_id = postings[i]
                        tf = tf_list[i]
                        dl = merged_index.doc_length[doc_id]

                        if doc_id not in scores:
                            scores[doc_id] = 0

                        if tf > 0:
                            # BM25 scoring
                            idf = math.log(N / df)
                            tf_component = (tf * (k1 + 1)) / \
                                (tf + k1 * (1 - b + b * (dl / avgdl)))
                            scores[doc_id] += idf * tf_component

            # Top-K retrieval
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
        SPIMI Indexing: Single-Pass In-Memory Indexing

        ALUR SPIMI:
        1. Baca documents block by block (single pass)
        2. Untuk setiap block, build in-memory hash map (no sorting of raw pairs)
        3. Dump block index ke disk ketika memory penuh atau block selesai
        4. Merge semua block indices menjadi final index

        PERBEDAAN dengan BSBI:
        - BSBI: Collect all (term,doc) pairs -> External Sort -> Build Index
        - SPIMI: Hash Map grouping -> Write sorted block -> Merge
        """
        # Pastikan output directory ada
        os.makedirs(self.output_dir, exist_ok=True)

        # Dapatkan semua blocks (sub-directories)
        try:
            blocks = sorted(next(os.walk(self.data_dir))[1])
        except StopIteration:
            print("No blocks found in data directory")
            return

        # Process each block (SPIMI: single pass through each block)
        for block_dir_relative in tqdm(blocks, desc="Indexing blocks"):
            # Parse block - single pass through documents
            td_pairs = self.parse_block(block_dir_relative)

            if not td_pairs:
                continue

            # Create intermediate index for this block
            index_id = 'intermediate_index_' + \
                block_dir_relative.replace('/', '_')
            self.intermediate_indices.append(index_id)

            # SPIMI: Build in-memory hash map and write to disk
            with InvertedIndexWriter(index_id, self.postings_encoding,
                                     directory=self.output_dir) as index:
                self.invert_write(td_pairs, index)

            # Clear memory
            td_pairs = None

        # Save term and doc mappings
        self.save()

        # Merge all intermediate indices into final index
        print("Merging intermediate indices...")
        with InvertedIndexWriter(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [
                    stack.enter_context(
                        InvertedIndexReader(index_id, self.postings_encoding,
                                            directory=self.output_dir)
                    )
                    for index_id in self.intermediate_indices
                ]
                self.merge(indices, merged_index)

        print(f"Indexing complete! Final index: {self.index_name}")


if __name__ == "__main__":
    SPIMI_instance = SPIMIIndex(
        data_dir='collection',
        postings_encoding=EliasGammaPostings,
        output_dir='index-spimi',
        memory_limit_mb=500  # 500 MB per block
    )
    SPIMI_instance.index()  # Start SPIMI indexing!
