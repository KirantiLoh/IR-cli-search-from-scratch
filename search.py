from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='collection',
                          postings_encoding=EliasGammaPostings,
                          output_dir='index')

queries = ["alkylated with radioactive iodoacetate",
           "psychodrama for disturbed children",
           "lipid metabolism in toxemia and normal pregnancy"]

for query in queries:
    print("Query  : ", query)
    print("Results:")
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=10):
        print(f"{doc:30} {score:>.3f}")
    print()
