print("Import time",flush=True)
import time
print("Import wns",flush=True)
start_time = time.time()
from WNS import Lemmatizer,sim_str_str,sim_str_str_multiling, toks_to_synsets, similarity_score, symetric_similarity_score
print("Time to import WNS",time.time() - start_time,"s")

#wc quijote.txt = 1876 words
txt = open("quijote.txt","r").read()
print("START PROGRAM",flush=True)
start_time = time.time()
lemmatizer = Lemmatizer(LANG="es",LANG_STOPWORDS="spanish")
print("Time to create Lemmatizer",time.time() - start_time,"s")
start_time = time.time()
lemmas = lemmatizer.lemmatize(txt)
end_time = time.time()
print("Time to lemmatize paragraph",end_time - start_time,"s")
print("    that is",(end_time - start_time)/len(lemmas),"seconds per lemma.")

#We measure how fast is synset recognition and path measurement
start_time = time.time()
toks_to_synsets(lemmas,lang="spa")
end_time = time.time()
print("Time to find synsets of the paragraph",end_time - start_time,"s")
print("    that is",(end_time - start_time)/len(lemmas),"seconds per lemma.")

toks1 = ["Hola","llamar","casa"]
s1 = toks_to_synsets(toks1,lang="spa")
toks2 = ["Adiós","volver","hotel"]
s2 = toks_to_synsets(toks2,lang="spa")
start_time = time.time()
symetric_similarity_score(s1,s2,stat="max")
end_time = time.time()
print("Time to measure similarity",end_time - start_time,"s")
print("    that is",(end_time - start_time)/(len(toks1)*len(toks1)),"seconds per pair of synsets.")


start_time = time.time()
print(sim_str_str("Estoy probando esta nueva aplicación en mi portátil conectado a internet.",
                    "Mi ordenador no funciona.",\
                    lang1="es",\
                    lang2="es",\
                    stat="max"))

print("Time to measure similarity without language inference",time.time() - start_time,"s")

start_time = time.time()
print(sim_str_str_multiling("Estoy probando esta nueva aplicación en mi portátil conectado a internet.",
                    "Mi ordenador no funciona.",\
                    stat="max"))
print("Time to measure similarity with language inference",time.time() - start_time,"s")

start_time = time.time()
print(sim_str_str("Estoy probando esta nueva aplicación en mi portátil conectado a internet.",
                    "Mi ordenador no funciona.",\
                    lang1="es",\
                    lang2="es",\
                    stat="max"))

print("Time to measure similarity without language inference",time.time() - start_time,"s")

start_time = time.time()
print(sim_str_str_multiling("Estoy probando esta nueva aplicación en mi portátil conectado a internet.",
                    "Mi ordenador no funciona.",\
                    stat="max"))
print("Time to measure similarity with language inference",time.time() - start_time,"s")