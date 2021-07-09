import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords  
import numpy as np
import pyfreeling
import re
import fasttext


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
        text = re.sub("_|\.|,|\"| etc|\(|\)|\||»|«|”|“|‘|’|[a-z-à-úïü]['’]|['’][a-z-à-úïü]"," ",text.lower())
        text = re.sub("•","·",text)
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

def ISO_6391_to_6392(code: str) -> str:
    """
    Converts ISO 639-1 (2 letters) language codes to ISO 639-2 (3 letters)
    """
    if code == "da":
        return "dan"
    elif code == "en":
        return "eng"
    elif code == "es":
        return "spa"
    elif code == "it":
        return "ita"
    elif code == "mo":
        return "mon"
    else:
        raise ValueError("ISO 639-1 code not known")

def meanList(l:list) -> float:
    if len(l)==0:
        return 0
    return sum(l)/len(l)

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
        
        if len(s1) == 0 or len(s2)==0:
            if s1 == s2:
                return 1
            else:
                return 0
        list1 = []

        count=0
        # For each synset in s1
        for a in s1:
            list2 = []
            for i in s2:
                # finds the synset in s2 with the largest similarity value
                score = i.path_similarity(a)
                if score is not None:
                    list2.append(score)
                else:
                    #If distance cannot be computed it is set to 0
                    list2.append(0)
            list1.append(max(list2))

        # print(list1)

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

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
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
        if len(syn)>0:
            synNames = []
            for s in syn:
                # s = s.name()
                output.extend(syn)
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


def sim_str_str(txt1: str, txt2: str, lemmatizer = Lemmatizer(),lang1="eng",lang2="eng",stat="max") -> float:
    toks1 = toks_to_synsets(lemmatizer.lemmatize(txt1),lang=lang1)    
    toks2 = toks_to_synsets(lemmatizer.lemmatize(txt2),lang=lang2)
    return symetric_similarity_score(toks1,toks2,stat=stat)

modelFasttext = fasttext.load_model('./lid.176.ftz')

def sim_str_str_multiling(txt1: str, txt2: str, lemmatizer = Lemmatizer(),stat="max") -> float:
    #We find out the language of the texts
    lang1 = modelFasttext.predict(txt1, k=1)[0][0][-2:] #We take the ISO code of the languages
    lang2 = modelFasttext.predict(txt2, k=1)[0][0][-2:]
    print(lang1)
    print(lang2)
    return sim_str_str(txt1,txt2,lemmatizer=lemmatizer,lang1=ISO_6391_to_6392(lang1),lang2=ISO_6391_to_6392(lang2),stat=stat)


# def sim_tokset_str():

if __name__ == '__main__':
    # print(sim_str_str("I am testing this new application on my laptop connected to the Internet.","My son studied computer science and he's working at Google",stat="max"))
    # print(sim_str_str_multiling("Estoy probando esta nueva aplicación en mi portátil conectado a internet.","My son studied computer science and he's working at Google",stat="max"))
    # print(sim_str_str_multiling("Mi padre irá a comprar jamón y queso a la tienda de la esquina.","My son studied computer science and he's working at Google",stat="max"))