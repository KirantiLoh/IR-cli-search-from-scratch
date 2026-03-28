import math
import re
from bsbi import BSBIIndex
from compression import EliasGammaPostings
from spimi import SPIMIIndex

# >>>>> sebuah IR metric: RBP p = 0.8


def rbp(ranking, p=0.8):
    """ menghitung search effectiveness metric score dengan 
        Rank Biased Precision (RBP)

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score RBP
    """
    score = 0.
    for i in range(1, len(ranking)):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def NDCG(ranking, p=0.8):
    """ menghitung search effectiveness metric score dengan 
        Normalized Discounted Cumulative Gain (NDCG)

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score NDCG
    """
    dcg = 0.
    for i in range(1, len(ranking)):
        pos = i - 1
        dcg += ranking[pos] / (math.log2(i + 1))

    # Hitung ideal DCG (IDCG)
    ideal_ranking = sorted(ranking, reverse=True)
    idcg = 0.
    for i in range(1, len(ideal_ranking) + 1):
        pos = i - 1
        idcg += ideal_ranking[pos] / (math.log2(i + 1))

    return dcg / idcg if idcg > 0 else 0.0


def dcg(ranking):
    """ menghitung Discounted Cumulative Gain (DCG) score

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score DCG
    """
    dcg_score = 0.
    for i in range(1, len(ranking)):
        pos = i - 1
        dcg_score += ranking[pos] / (math.log2(i + 1))
    return dcg_score


def AP(ranking):
    """ menghitung Average Precision (AP) score

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score AP
    """
    relevant_count = 0
    precision_sum = 0.0

    for i in range(1, len(ranking)):
        pos = i - 1
        if ranking[pos] == 1:
            relevant_count += 1
            precision_sum += relevant_count / i

    return precision_sum / relevant_count if relevant_count > 0 else 0.0


# >>>>> memuat qrels

def load_qrels(qrel_file="qrels.txt", max_q_id=30, max_doc_id=1033):
    """ memuat query relevance judgment (qrels) 
        dalam format dictionary of dictionary
        qrels[query id][document id]

        dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
        relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
        Doc 10 tidak relevan dengan Q3.

    """
    qrels = {"Q" + str(i): {i: 0 for i in range(1, max_doc_id + 1)}
             for i in range(1, max_q_id + 1)}
    with open(qrel_file) as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = int(parts[1])
            qrels[qid][did] = 1
    return qrels

# >>>>> EVALUASI !


def eval(qrels, query_file="queries.txt", k=1000):
    """ 
      loop ke semua 30 query, hitung score di setiap query,
      lalu hitung MEAN SCORE over those 30 queries.
      untuk setiap query, kembalikan top-1000 documents
    """
    BSBI_instance = SPIMIIndex(data_dir='collection',
                               postings_encoding=EliasGammaPostings,
                               output_dir='index', wand_config={"use_wand": True, "scoring_function": "bm25"})

    # PERBAIKAN 1: Baca semua query ke dalam list terlebih dahulu
    with open(query_file) as file:
        queries = file.readlines()

    for scoring_function in ["bm25", "tfidf", "wand_optimized"]:
        rbp_scores = []
        ndcg_scores = []
        dcg_scores = []
        ap_scores = []

        print(f"Evaluating with scoring function: {scoring_function}")

        # PERBAIKAN 2: Iterasi menggunakan list 'queries', bukan object file
        for qline in queries:
            parts = qline.strip().split()
            if not parts:
                continue

            qid = parts[0]
            query = " ".join(parts[1:])

            ranking = []
            # Ekstraksi doc ID bisa digabung atau dipisah, pastikan regex cocok
            # Saya tambahkan pengecekan jika regex gagal (None) agar tidak crash
            if scoring_function == "bm25":
                for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k):
                    match = re.search(r'\/.*\/.*\/(.*)\.txt', doc)
                    if match:
                        did = int(match.group(1))
                        ranking.append(qrels[qid][did])
            elif scoring_function == "tfidf":
                for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=k):
                    match = re.search(r'\/.*\/.*\/(.*)\.txt', doc)
                    if match:
                        did = int(match.group(1))
                        ranking.append(qrels[qid][did])
            elif scoring_function == "wand_optimized":
                for (score, doc) in BSBI_instance.retrieve_wand_optimized(query, k=k, scoring="bm25"):
                    match = re.search(r'\/.*\/.*\/(.*)\.txt', doc)
                    if match:
                        did = int(match.group(1))
                        ranking.append(qrels[qid][did])

            rbp_scores.append(rbp(ranking))
            ndcg_scores.append(NDCG(ranking))
            dcg_scores.append(dcg(ranking))
            ap_scores.append(AP(ranking))

        print(
            f"Hasil evaluasi {scoring_function} terhadap {len(rbp_scores)} queries")

        # PERBAIKAN 3: Defensive check agar tidak division by zero jika list tetap kosong
        if len(rbp_scores) > 0:
            print("RBP score =", sum(rbp_scores) / len(rbp_scores))
            print("NDCG score =", sum(ndcg_scores) / len(ndcg_scores))
            print("DCG score =", sum(dcg_scores) / len(dcg_scores))
            print("AP score =", sum(ap_scores) / len(ap_scores))
        else:
            print(
                "Tidak ada query yang diproses (Check query file atau retrieval method).")


if __name__ == '__main__':
    qrels = load_qrels()

    assert qrels["Q1"][166] == 1, "qrels salah"
    assert qrels["Q1"][300] == 0, "qrels salah"

    eval(qrels)
