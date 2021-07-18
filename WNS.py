import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from nltk.corpus import stopwords  
import numpy as np
import pyfreeling
import re
import fasttext
import marshal
from alive_progress import alive_bar


class Lemmatizer:

    def __init__(self,
                DATA="/usr/local/share/freeling/",
                LANG="en",
                LANG_STOPWORDS="english"):
        self.stop_words = set(stopwords.words(LANG_STOPWORDS))
        self.DATA = DATA
        pyfreeling.util_init_locale("default")
        self.LANG=LANG
        self.op= pyfreeling.maco_options(LANG)
        self.op.set_data_files( "",
                    DATA + "common/punct.dat",
                    DATA + LANG + "/dicc.src",
                    DATA + LANG + "/afixos.dat",
                    "",
                    DATA + LANG + "/locucions.dat",
                    DATA + LANG + "/np.dat",
                    DATA + LANG + "/quantities.dat",
                    DATA + LANG + "/probabilitats.dat")

        self.sp=pyfreeling.splitter(DATA+LANG+"/splitter.dat")
        self.sid=self.sp.open_session()
        self.mf=pyfreeling.maco(self.op)
        # activate mmorpho modules to be used in next call
        self.mf.set_active_options(umap=False, num=True, pun=True, dat=False,  # select which among created
                            dic=True, aff=True, comp=False, rtk=True,  # submodules are to be used.
                            mw=True, ner=True, qt=True, prb=True ); # default: all created submodules are used
        self.tk=pyfreeling.tokenizer(DATA+LANG+"/tokenizer.dat");
    
    def lemmatize(self,text:str):
        #First we remove some special characters
        # text = re.sub("_|\.|:|,|\"| etc|\(|\)|\||»|«|”|“|‘|’|[a-z-à-úïü]['’]|['’][a-z-à-úïü]"," ",text.lower())
        text = re.sub("•","·",text.lower())
        text = re.sub("l.l","l·l",text)

        #Freeling's Splitter needs a EOF mark or it will fail, that's why we put a final dot
        if text[-1]!=".":
                text=text+"."
        l = self.tk.tokenize(text);
        ls = self.sp.split(self.sid,l,False);
        ls = self.mf.analyze(ls)

        lemmas = []
        for s in ls:
            for w in s:
                lemmas.append(w.get_lemma())
        
        res = [l for l in lemmas if ((l!=".") and (l not in self.stop_words))] 
        # print(res)
        return res
        # return  [l for l in lemmas if ((l!=".") and (l not in stop_words))] 

def meanList(l:list) -> float:
    if len(l)==0:
        return 0
    return sum(l)/len(l)


langs_iso_6291 = set(["ca","da","en","es","it","mn"])

modelFasttext = fasttext.load_model('./lid.176.bin')

scores = {}
try:
    f = open("./scores.marshal","rb")
    scores = marshal.load(f)
    f.close()
except IOError:
    scores = {}
print(len(scores))


def fill_scores(text: str, lang: str) -> dict:
    lemmatizer = Lemmatizer(LANG=lang,LANG_STOPWORDS=ISO_6391_to_name(lang))
    lemmas = lemmatizer.lemmatize(text)
    lemmas = set(lemmas)
    lemmas_cp = lemmas.copy()
    langISO6392 = ISO_6391_to_6392(lang)
    synsets = []
    print("Lemmatize")
    with alive_bar(len(lemmas),force_tty=1) as bar:
        for l in lemmas:
            bar()
            synset = toks_to_synsets([l],lang=langISO6392)
            if synset == []:
                lemmas_cp.remove(l)
            else:
                synsets.append(synset[0])
    lemmas = lemmas_cp.copy()
    new_scores = {}
    print("Compute similarity")
    with alive_bar(len(synsets)**2,force_tty=1) as bar:
        for i,s1 in enumerate(synsets):
            for j,s2 in enumerate(synsets):
                bar()
                if j<i:
                    continue
                else:
                    score = s1.path_similarity(s2)
                    new_scores[s1.name()+s2.name()] = score
    foutput = open("./scores.marshal","wb")
    marshal.dump(new_scores,foutput)
    foutput.close()

def ISO_6391_to_6392(code: str) -> str:
    """
    Converts ISO 639-1 (2 letters) language codes to ISO 639-2 (3 letters)
    """
    if code == "ca":
        return "cat"
    if code == "da":
        return "dan"
    elif code == "en":
        return "eng"
    elif code == "es":
        return "spa"
    elif code == "it":
        return "ita"
    elif code == "mn":
        return "mon"
    else:
        raise ValueError("ISO 639-1 code not known: "+str(code))

def ISO_6391_to_name(code: str) -> str:
    """
    Converts ISO 639-1 (2 letters) language codes to common name (eg: "en" -> "english")
    """
    if code == "ca":
        return "catalan"
    if code == "da":
        return "danish"
    elif code == "en":
        return "english"
    elif code == "es":
        return "spanish"
    elif code == "it":
        return "italian"
    elif code == "mn":
        return "mongolian"
    else:
        raise ValueError("ISO 639-1 code not known: "+str(code))


def similarity_score(s1, s2, stat = "max"):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    
    if len(s1) == 0 or len(s2)==0 or s1==None or s2==None:
        return 0
    list1 = []

    count=0
    # For each synset in s1
    for a in s1:
        list2 = []
        for i in s2:
            if a.name()+i.name() in scores:
                list2.append(scores[a.name()+i.name()])
            else:
                # finds the synset in s2 with the largest similarity value
                score = i.path_similarity(a)
                if score is not None:
                    list2.append(score)
                else:
                    #If distance cannot be computed it is set to 0
                    list2.append(0)
        list1.append(max(list2))

    if stat == "max":
        output = max(list1)
    elif stat == "mean":
        output = meanList(list1)
    elif stat == "q75":
        output = np.quantile(np.array(list1),0.75)
    elif stat == "q90":
        output = np.quantile(np.array(list1),0.90)
    else:
        raise ValueError("Stat still not suported")
    return output


def symetric_similarity_score(s1, s2, stat = "max"):
    return (similarity_score(s1, s2,stat) + similarity_score(s2, s1,stat)) / 2


def toks_to_synsets(toks, pos = None, lang = "eng"):
    """
    Returns a list of synsets in a list of tokens.

    Then finds all the synsets for each word combination.
    If a synset is not found for that combination it is skipped.

    Args:
        toks: List of tokens to be converted
        pos: Whether to use PoS info or leave it as None

    Returns:
        list of synsets

    Example:
        toks_to_synsets(['Fish', 'are', 'nvqjp', 'friends'])
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """

    output = []
    for i in toks:
        syn = wn.synsets(i,pos=None,lang=lang)
        syn = syn[0:min(1,len(syn))]
        if len(syn)>0:
            synNames = []
            for s in syn:
                # s = s.name()
                output.extend(syn)
    # print(output)
    return output


def tokLists_path_similarity(self,tokLists1, tokLists2, stat="max"):
    """Finds the symmetrical similarity between two lists 
    of lists of tokens (two lists of documents)"""
            # first function u need to create
    synsetsLists1 = self.tokLists_to_synsets(tokLists1)
    synsetsLists2 = self.tokLists_to_synsets(tokLists2)
            # 2nd function u need to create

    with alive_bar(len(synsetsLists1)*len(synsetsLists2),force_tty=1) as bar:
        sims = np.full(fill_value=-1.0,shape=(len(tokLists1),len(tokLists2)))
        for s1 in range(len(synsetsLists1)):
            for s2 in range(len(synsetsLists2)):
                bar()
                sims[s1,s2] = (self.similarity_score(synsetsLists1[s1], synsetsLists2[s2],stat) + self.similarity_score(synsetsLists2[s2], synsetsLists1[s1],stat)) / 2

    return sims


def sim_str_str(txt1: str, txt2: str, lemmatizer1 = Lemmatizer(),lemmatizer2 = Lemmatizer(),lang1="eng",lang2="eng",stat="max") -> float:
    """
    Finds the symetric similarity score between two texts, aggregating the
    path similarity of the synsets according the stat argument.

    Parameters
    ----------
    txt1: str
        First of the two texts.
    txt2: str
        Second of the two texts.
    lemmatizer1: Lematizer
        The Lemmatizer used to find the lemmas of the tokens in text1.
    lemmatizer2: Lematizer
        The Lemmatizer used to find the lemmas of the tokens in text2.
    lang1: str
        Language of the first text in ISO 639-2 format.
    lang2: str
        Language of the second text in ISO 639-2 format.
    stat: str
        Statistical function to aggregate the similarity between lemmas.
    """
    toks1 = toks_to_synsets(lemmatizer1.lemmatize(txt1),lang=lang1)    
    toks2 = toks_to_synsets(lemmatizer2.lemmatize(txt2),lang=lang2)
    return symetric_similarity_score(toks1,toks2,stat=stat)




def sim_str_str_multiling(txt1: str, txt2: str,stat="max") -> float:
    #We find out the language of the texts
    lang1 = modelFasttext.predict(txt1, k=10)[0] #We take the ISO code of the languages
    # print(lang1)
    lang1 = next(l[-2:] for l in lang1 if l[-2:] in langs_iso_6291)
    lemmatizer1 = Lemmatizer(LANG=lang1, LANG_STOPWORDS=ISO_6391_to_name(lang1))
    lang2 = modelFasttext.predict(txt2, k=10)[0]
    # print(lang2)
    lang2 = next(l[-2:] for l in lang2 if l[-2:] in langs_iso_6291)
    lemmatizer2 = Lemmatizer(LANG=lang2, LANG_STOPWORDS=ISO_6391_to_name(lang2))
    return sim_str_str(txt1,txt2,lemmatizer1=lemmatizer1,lemmatizer2=lemmatizer2,lang1=ISO_6391_to_6392(lang1),lang2=ISO_6391_to_6392(lang2),stat=stat)


# def sim_tokset_str():
