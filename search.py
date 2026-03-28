from bsbi import BSBIIndex
from spimi import SPIMIIndex
from compression import VBEPostings, EliasGammaPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = SPIMIIndex(data_dir='collection',
                           postings_encoding=EliasGammaPostings,
                           output_dir='index-spimi', wand_config={"use_wand": True, "scoring_function": "bm25"})
# BSBI_instance = BSBIIndex(data_dir='collection',
#                           postings_encoding=EliasGammaPostings,
#                           output_dir='index')

queries = ["alkylated with radioactive iodoacetate",
           "psychodrama for disturbed children",
           "lipid metabolism in toxemia and normal pregnancy"]

for query in queries:
    print("Query  : ", query)
    print("Results:")
    for (score, doc) in BSBI_instance.retrieve_wand_optimized(query, k=10):
        print(f"{doc:30} {score:>.3f}")
    print()
