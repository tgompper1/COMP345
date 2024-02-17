from collections import Counter
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.matutils import unitvec
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import cosine_similarity
import re
import math 

"""
Tess Gompper #260947251
https://docs.google.com/document/d/1IGcdsRUJ_i17SG7Iqw0vSOGmJ8KqAnlaZGY0umdcu5Q/edit?usp=sharing 
"""
PATTERN = re.compile("^[a-z#]")

def top_k_unigrams(tweets, stop_words, k):
    """
    A helper function that is used to determine the most frequently occurring words in the corpus. 

    tweets : list of str
        Cleaned and tokenized tweets from Assignment 1
    stop_words : list of str
        Words not to be considered while deciding most frequent words
    k : int
        Number of most frequent words to return, along with their counts. 
        If k is -1, then return the entire count dictionary without any filtering.

    return: top_k_words : dict of {str: int}
        A dictionary of top-k words with their frequency counts. 
        It will have all words with their counts if k=-1
    """ 
    all_tokens = []
    for tweet in tweets:
        tokens = tweet.split() 
        all_tokens.extend(tokens)

    counts = {}
    for token in all_tokens:
        if token in stop_words:
            continue
        elif PATTERN.match(token):
            if token in counts:
                counts[token] += 1
            else:
                counts[token] = 1

    if k == -1:
        return counts
    else:
        sorted_counts = sorted(counts.items(), key=lambda x:x[1], reverse=True)
        sorted_counts_dict = dict(sorted_counts)
        top_k_words = dict(list(sorted_counts_dict.items())[0: k])

        return top_k_words

def context_word_frequencies(tweets, stop_words, context_size, frequent_unigrams):
    """
    A helper function that is used to get the frequency of pairs of words that will eventually 
        be used for PMI calculations. 

    tweets : list of str
        Cleaned and tokenized tweets from Assignment 1
    stop_words : list of str
        Words not to be considered while deciding most frequent words
    context_size : int
        Number of words to be considered in context on left and right
    frequent_unigrams : list of str
        Top-1000 Frequent unigrams computed in the previous function (converted to a list)

    return: context_counter : dict of {(str, str): int}
        A dictionary of frequencies of pairs of words
    """ 
    context_counter = {}
    lower_context = -1*context_size
    upper_context = context_size + 1
    for tweet in tweets:
        tokens = tweet.split()
        l = len(tokens)
        for i in range(l):
            word_1 = tokens[i]

            for j in range(lower_context, upper_context):
                if j == 0:
                    continue
                elif i+j >= 0 and i+j < l:
                    word_2 = tokens[i+j]
                    if word_2 not in stop_words and PATTERN.match(word_2) and word_2 in frequent_unigrams:
                        if (word_1, word_2) in context_counter:
                            context_counter[(word_1, word_2)] += 1
                        else:
                            context_counter[(word_1, word_2)] = 1
    return context_counter
            
def pmi(word1, word2, unigram_counter, context_counter):
    """
    A helper function to calculate Pointwise Mutual Information for a pair of words 
        based on unigram and bigram counts. If the unigram or bigram has not been observed,
        assume a pseudo count of 1. However, the total count of unigrams will be fixed and 
        will be independent of whether the actual count is used or the pseudo count.
    
    word_1 : str
        1st word of the pair for which PMI has to be calculated
    word_2 : str
        2nd word of the pair for which PMI has to be calculated
    unigram_counter : dict of {str: int}
        All words in the corpus along with their frequencies
    context_counter : dict of {(str, str): int}
        Frequencies of pairs of words as computed in the previous function.
    
    return: pmi : float
        PMI value for word_1 and word_2
    """
    psuedo_count = 1

    if word1 in unigram_counter:
        freq_word_1 = unigram_counter[word1]
    else:
        freq_word_1 = psuedo_count
    if word2 in unigram_counter:
        freq_word_2 = unigram_counter[word2]
    else:
        freq_word_2 = psuedo_count
    N = sum(unigram_counter.values())
    exp_freq = (freq_word_1/N) * (freq_word_2/N)
    if (word1, word2) in context_counter:
        obs = context_counter[(word1, word2)]
    else:
        obs = 1
    obs_freq = obs/N
    pmi = np.log2(obs_freq/exp_freq)
    
    return pmi

def build_word_vector(word1, frequent_unigrams, unigram_counter, context_counter):
    """
    A helper function to construct a word vector based on PMI values using statistics 
        computed from previous functions. The top k frequent words define the dimensions
        of the vector space model.
    
    word_1 : str 
        Word for which word vector has to be computed
    frequent_unigrams : list of str
        Top-k frequent unigrams
    unigram_counter : dict of {str: int}
        All words in the corpus along with their frequencies
    context_counter : dict of {(str, str): int}
        Frequencies of pairs of words.
    
    return: word_vector : dict of {str: float}
        Word vector of word_1 represented as a dictionary
    """
    word_vector = {}

    for word in frequent_unigrams:
        if (word1, word) not in context_counter:
            score = float(0)
        else:
            score = pmi(word1, word, unigram_counter, context_counter)
            
        word_vector[word] = score

    return word_vector

def get_top_k_dimensions(word1_vector, k):
    """
    This helper functions helps to visualize the dimensions (and by extension context words) 
        most relevant to a given word, using its vector. The higher the value of a particular 
        dimension in the word vector, the greater the relevance
    
    word1_vector : dict of {str: float}
        Word vector
    k : int 
        Top-k dimensions to be returned

    return:  top_k_dimensions : dict of {str: float}
        Top-k dimensions of the input word vector based on dimension values.
    """
    sorted_vector = sorted(word1_vector.items(), key=lambda x:x[1], reverse=True)
    sorted_vector_dict = dict(sorted_vector)
    top_k_dimensions = dict(list(sorted_vector_dict.items())[0: k])

    return top_k_dimensions

def get_cosine_similarity(word1_vector, word2_vector):
    """
    Given word vectors of two words, their relatedness is measured using 
        cosine similarity.

    word1_vector : dict of {str: float}
        Word vector of the 1st word
    word2_vector : dict of {str: float}
        Word vector of the 2nd word

    return: cosine_sim_score : float
        Cosine similarity between the two word vectors
   
    """
    dot = sum(word1_vector.get(word, 0) * word2_vector.get(word, 0) for word in set(word1_vector) & set(word2_vector))

    mag1 = math.sqrt(sum(value**2 for value in word1_vector.values()))
    mag2 = math.sqrt(sum(value**2 for value in word2_vector.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    cosine_sim_score = dot / (mag1 * mag2)

    return cosine_sim_score

def get_most_similar(word2vec, word, k):
    """
    Given a pre-loaded word2vec model and a word, the function should return k
        most similar words from the vocabulary based on the cosine similarity.
        If the word is not in the vocabulary, the function should return an empty list
    
    word2vec : gensim.models.keyedvectors.KeyedVectors
        Pre-loaded word2vec model from gensim
    word : str
        Word for which similar words have to be found
    k : int
        Number of similar words to return
    
    return: similar_words : List of (str, float)
        List of similar words along with cosine similarity score.
    """
    if word not in word2vec:
        return []
    else:
        similar_words = word2vec.most_similar(word, topn=k)
        return similar_words

def word_analogy(word2vec, word1, word2, word3):
    """
    Word analog in this context means answering the question: X is to Y as A is to what?
        For example, Tokyo is to Japan as Paris is to what?. The answer is France
        Input to this function will be three words. The order of these three words 
        is very important. Be careful. For the analogy task, X is to Y as A is to what? 
        The first input is X, the second input is Y and the third input is A.

    word2vec : gensim.models.keyedvectors.KeyedVectors
        Pre-loaded word2vec model from gensim
    word1 : str 
        X
    word2: str 
        Y
    word3 : str 
        A

    return: word_4 : str
        Word that is the answer to the analogy task, along with its score as given
        by word2vec model.
    """
    analogous = word2vec.most_similar(positive=[word2, word3], negative=[word1], topn=1)

    if analogous:
        return analogous[0]
    else:
        return None

def cos_sim(A, B):
    """
    A function that returns the cosine similarity of two numerical vectors.
        Importantly, unlike the get_cosine_similarity function, this function 
        takes as its inputs two Numpy arrays or lists of numerical values,
        rather than two dictionaries of {str: float}.
    
    A : np.ndarray or list of numerical values
        The first vector to be compared.
    B : np.ndarray or list of numerical values
        The other vector to be compared.

    return: cosine_similarity : float
        The cosine similarity of vectors A and B.
    """
    # in case vectors are not numpy arrays
    A = np.array(A)
    B = np.array(B)

    norm_A = unitvec(A)
    norm_B = unitvec(B)

    dot = np.dot(norm_A, norm_B)

    return dot

def get_cos_sim_different_models(word, model1, model2, cos_sim_function):
    """
    A function that returns the cosine similarity of the embeddings of a 
        given word from two different models. This is one way of looking at 
        how the meaning of a word has changed over time: taking word embeddings, 
        for the same word, from two models trained on different time periods’ 
        text, and comparing their similarity.

    word : str
        The target word, whose embeddings from different models the function 
        compute cosine similarity between.
    model1 : gensim.models.word2vec.Word2Vec
        The first Word2Vec model whose embedding of the target word will be compared
    model2 : gensim.models.word2vec.Word2Vec
        The other Word2Vec model whose embedding of the target word will be compared.
    cos_sim_function : function
        The function used to compute cosine similarity between two vectors.
    
    return: cosine_similarity_of_embeddings : float
        Cosine similarity between the embeddings of the target word from the 
        first and second Word2Vec model. 
    """
    embedding1 = model1.wv[word]
    embedding2 = model2.wv[word]

    dot = cos_sim_function(embedding1, embedding2)

    return dot

def get_average_cos_sim(word, neighbors, model, cos_sim_function):
    """
    a function that returns the average of the cosine similarities between
        the embeddings of a target word and the embeddings of a set of 
        ‘neighborhood’ words. This is another way of looking at how the
        meaning of a word has changed over time, without worrying about 
        alignment: by comparing how similar a word’s embedding is to those
        of different sets of words at different points in time. In cases 
        where not all of the neighbors (words) have embeddings in a model
        the function should ignore those words that do not have an embedding
        in the model, and return an average similarity based only on the
        available embeddings.

    word : str
        The target word, whose embedding the function compares with those
        of ‘neighborhood’ words.
    neighbors : List of str
        A list of words whose embeddings the target word’s embedding will 
        be compared with.
    model : gensim.models.word2vec.Word2Vec
        The Word2Vec model whose embeddings will be used.
    cos_sim_function : function
        The function used to compute cosine similarity between two vectors

    return: avg_cosine_similarity : float 
        Mean of the cosine similarities between the embedding of the target
        word and the embedding of each of the ‘neighbor’ words.
    """
    total = 0
    count = 0

    embedding = model.wv[word]

    for neighbor in neighbors:
        if neighbor not in model.wv.key_to_index:
            continue
        else:
            neighbor_embedding = model.wv[neighbor]
                
            total += cos_sim_function(embedding, neighbor_embedding)
            count += 1
    
    avg_cosine_similarity  = total / max(count, 1)

    return avg_cosine_similarity 

def create_tfidf_matrix(documents, stopwords):
    """
    Given a list of documents and a list of stopwords, construct the 
        TF-IDF matrix for the corpus. Using NLTK stopwords. Documents 
        are pre-processed in the following manner -
        1) lowercasing words, 
        2) excluding stopwords, and 
        3) including alphanumeric strings only (use isalnum). 
        All the words remaining in the documents after pre-processing 
        will constitute the vocabulary. The function returns the TF-IDF 
        matrix of dimension (num_docs, num_words) and the vocabulary as 
        inferred from the corpus. The vocabulary returned is sorted
        in alphabetical order. The dimensions of the TF-IDF matrix match
        the order of the documents and the order of the sorted vocabulary. 
        i.e. (i,j)cell of the matrix should denote the TF-IDF score of the
        ith document and jth word in the sorted vocabulary
    
    documents : list of nltk.corpus.reader.tagged.TaggedCorpusView
        Collection of documents using NLTK corpuses
    stopwords : list of str
        List of stopwords

    return: tfidf_matrix : np.ndarray
        TF-IDF matrix
    vocab : list of str
        List of vocabulary terms as inferred from the corpus
    """
    # pre-process documents
    preprocessed_docs = []
    for document in documents:
        preprocessed_doc = [word.lower() for word in document if word.lower() not in stopwords]
        preprocessed_doc = [word for word in preprocessed_doc if word.isalnum()]
        preprocessed_docs.append(preprocessed_doc)

    vocab = sorted(set(word for document in preprocessed_docs for word in document))

    # Raw count term frequency
    tf_matrix = np.zeros((len(documents), len(vocab)))
    for i, doc in enumerate(preprocessed_docs):
        word_counts = Counter(doc)
        for j, word in enumerate(vocab):
            tf_matrix[i, j] = word_counts[word]

    df = np.sum(tf_matrix > 0, axis=0)

    # smoothened IDF
    n = len(documents)
    idf = np.log10(n / (1 + df)) + 1

    tfidf_matrix = tf_matrix * idf

    return tfidf_matrix, vocab

def get_idf_values(documents, stopwords):
    # pre-process documents
    preprocessed_docs = []
    for document in documents:
        preprocessed_doc = [word.lower() for word in document if word.lower() not in stopwords]
        preprocessed_doc = [word for word in preprocessed_doc if word.isalnum()]
        preprocessed_docs.append(preprocessed_doc)

    vocab = sorted(set(word for document in preprocessed_docs for word in document))

    # Raw count term frequency
    tf_matrix = np.zeros((len(documents), len(vocab)))
    for i, doc in enumerate(preprocessed_docs):
        word_counts = Counter(doc)
        for j, word in enumerate(vocab):
            tf_matrix[i, j] = word_counts[word]

    df = np.sum(tf_matrix > 0, axis=0)

    # smoothened IDF
    n = len(documents)
    idf = np.log10(n / (1 + df)) + 1
    
    vocab_idf_dict = dict(zip(vocab, idf))
    
    return vocab_idf_dict

def calculate_sparsity(tfidf_matrix):
    """
    TF-IDF matrices are generally extremely sparse. This function,
        given a TF-IDF matrix, calculates the ratio between cells with 
        value 0 and the total number of cells. 
    
    tfidf_matrix : np.ndarray
        TF-IDF matrix as computed in the previous function
    
    return: sparsity : float
        Sparsity of the matrix, detonated as a value between 0 and 1
    """
    total_cells = tfidf_matrix.size
    zero_cells = np.sum(tfidf_matrix == 0)
    sparsity = zero_cells / total_cells
    return sparsity

def extract_salient_words(VT, vocabulary, k):
    """
    It is possible to understand the hidden dimensions of the LSA model
        by computing words from the vocabulary that are most associated
        with those dimensions. For each of the 10 latent dimensions of LSA,
        this function finds the top k most relevant terms. Dimensions are
        ordered as 0, 1, 2, …

    VT : np.ndarray
        VT from DVD decomposition of TF-IDF matrix
    vocabulary : list of str 
        Vocabulary computed from corpus
    k : int
        Number of top salient words to return for each dimension
    
    return: salient_words : dict of {int: list of str}
        Top k salient words for each hidden dimension
    """
    num_dimensions = 10
    salient_words = {}

    for dim in range(num_dimensions):
        dim_row = VT[dim, :]

        top_k = np.argsort(dim_row)[-k:][::-1]

        terms = list(np.array(vocabulary)[top_k])
        terms.reverse()

        salient_words[dim] = terms

    return salient_words

def get_similar_documents(U, Sigma, VT, doc_index, k):
    """
    Given a document index, return the indices of top k similar documents,
        excluding the input document itself.

    U : np.ndarray
        U from SVD decomposition of TF-IDF matrix
    Sigma : np.ndarray
        Sigma from SVD decomposition of TF-IDF matrix
    VT : np.ndarray
        VT from SVD decomposition of TF-IDF matrix
    doc_index : int
        Index of the document for which similar documents have to be retrieved.
        This refers to the doc_index row in the original TF-IDF matrix.
    k : int
        Number of top similar documents to return
    
    return: similar_doc_indices : list of int
        Indices of similar documents in the collection
    
    https://en.wikipedia.org/wiki/Latent_semantic_analysis
    """
    X = np.matmul(np.matmul(U, np.diag(Sigma)), VT)
    doc_vec = X[doc_index, :]

    cos_similarities = cosine_similarity([doc_vec], X)[0]
    similar_documents_indices = np.argsort(cos_similarities)[::-1][1:k+1] ## start at 1 because it's most similar to itself

    return list(similar_documents_indices)

def document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, k):
    """
    Given a query, first, create a TF-IDF representation of the query. 
        Then project it into the low-dimensional LSA semantic space. Finally,
        compute the similarity with all the documents in the corpus and return
        the indices of top k

    vocabulary : list of str
        Vocabulary computed from corpus
    idf_values : dict of {str: float}
        Words from vocabulary along with their IDF values
    U : np.ndarray
        U from SVD decomposition of TF-IDF matrix
    Sigma : np.ndarray
        Sigma from SVD decomposition of TF-IDF matrix
    VT : np.ndarray
        VT from SVD decomposition of TF-IDF matrix
    query : list of str
        Query for which top documents have to be retrieved
    k : int
        Number of top documents indices to return
    
    return: retrieved_doc_indices : list of int
        Indices of documents relevant to the query in decreasing order of 
        relevance score.
    """  
    tokens = list(map(lambda x:x.lower(), query))
    
    query_vector= np.zeros((len(vocabulary)))
    query_counter = Counter(tokens)

    for i, term in enumerate(vocabulary):
        if term in tokens:
            query_vector[i] = (query_counter[term]) * idf_values[term]

    X = np.matmul(np.matmul(U, np.diag(Sigma)), VT)

    cos_similarities = cosine_similarity([query_vector], X)[0]
    similar_documents_indices = np.argsort(cos_similarities)[::-1][0:k] ## this goes from 0 not 1 (as above) because the query is not in the set of documents

    return list(similar_documents_indices)

if __name__ == '__main__':
    
    tweets = []
    with open('data/covid-tweets-2020-08-10-2020-08-21.tokenized.txt', encoding="utf8") as f:
        tweets = [line.strip() for line in f.readlines()]
    
    stop_words = []
    with open('data/stop_words.txt') as f:
        stop_words = [line.strip() for line in f.readlines()]


    """Building Vector Space model using PMI"""

    print(top_k_unigrams(tweets, stop_words, 10))
    # {'covid': 71281, 'pandemic': 50353, 'covid-19': 33591, 'people': 31850, 'n’t': 31053, 'like': 20837, 'mask': 20107, 'get': 19982, 'coronavirus': 19949, 'trump': 19223}
    frequent_unigrams = top_k_unigrams(tweets, stop_words, 1000)
    unigram_counter = top_k_unigrams(tweets, stop_words, -1)
  
    ### THIS PART IS JUST TO PROVIDE A REFERENCE OUTPUT
    sample_output = context_word_frequencies(tweets, stop_words, 2, frequent_unigrams)
   
    print(sample_output.most_common(10))
    """
    [(('the', 'pandemic'), 19811),
    (('a', 'pandemic'), 16615),
    (('a', 'mask'), 14353),
    (('a', 'wear'), 11017),
    (('wear', 'mask'), 10628),
    (('mask', 'wear'), 10628),
    (('do', 'n’t'), 10237),
    (('during', 'pandemic'), 8127),
    (('the', 'covid'), 7630),
    (('to', 'go'), 7527)]
    """
    ### END OF REFERENCE OUTPUT
    
    context_counter = context_word_frequencies(tweets, stop_words, 3, frequent_unigrams)
    
    word_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'put': 6.301874856316369, 'patient': 6.222687002250096, 'tried': 6.158108051673095, 'wearing': 5.2564459708663875, 'needed': 5.247669358807432, 'spent': 5.230966480014661, 'enjoy': 5.177980198384708, 'weeks': 5.124941187737894, 'avoid': 5.107686157639801, 'governors': 5.103879572210065}

    word_vector = build_word_vector('mask', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'wear': 7.278203356425305, 'wearing': 6.760722107602916, 'mandate': 6.505074539073231, 'wash': 5.620700962265705, 'n95': 5.600353617179614, 'distance': 5.599542578641884, 'face': 5.335677912801717, 'anti': 4.9734651502193366, 'damn': 4.970725788331299, 'outside': 4.4802694058646}

    word_vector = build_word_vector('distancing', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'social': 8.637723567642842, 'guidelines': 6.244375965192868, 'masks': 6.055876420939214, 'rules': 5.786665161219354, 'measures': 5.528168931193456, 'wearing': 5.347796214635814, 'required': 4.896659865603407, 'hand': 4.813598338358183, 'following': 4.633301876715461, 'lack': 4.531964710683777}

    word_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'donald': 7.363071158640809, 'administration': 6.160023745590209, 'president': 5.353905139926054, 'blame': 4.838868198365827, 'fault': 4.833928177006809, 'calls': 4.685281547339574, 'gop': 4.603457978983295, 'failed': 4.532989597142956, 'orders': 4.464073158650432, 'campaign': 4.3804665561680824}

    word_vector = build_word_vector('pandemic', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'global': 5.601489175269805, 'middle': 5.565259949326977, 'amid': 5.241312533124981, 'handling': 4.609483077248557, 'ended': 4.58867551721951, 'deadly': 4.371399989758025, 'response': 4.138827482426898, 'beginning': 4.116495953781218, 'pre': 4.043655804452211, 'survive': 3.8777495603541254}

    word1_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('covid-19', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.2341567704935342

    word2_vector = build_word_vector('mask', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.05127326904936171

    word1_vector = build_word_vector('president', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.7052644362543867

    word2_vector = build_word_vector('biden', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.6144272810573133

    word1_vector = build_word_vector('trudeau', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.37083874436657593

    word2_vector = build_word_vector('biden', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.34568665086152817


    """Exploring Word2Vec"""

    EMBEDDING_FILE = 'data/GoogleNews-vectors-negative300.bin.gz'
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

    similar_words =  get_most_similar(word2vec, 'ventilator', 3)
    print(similar_words)
    # [('respirator', 0.7864563465118408), ('mechanical_ventilator', 0.7063839435577393), ('intensive_care', 0.6809945702552795)]

    # Word analogy - Tokyo is to Japan as Paris is to what?
    print(word_analogy(word2vec, 'Tokyo', 'Japan', 'Paris'))
    # ('France', 0.7889978885650635)


    # """Word2Vec for Meaning Change"""

    # Comparing 40-60 year olds in the 1910s and 40-60 year olds in the 2000s
    model_t1 = Word2Vec.load('data/1910s_50yos.model')
    model_t2 = Word2Vec.load('data/2000s_50yos.model')

    # Cosine similarity function for vector inputs
    # vector_1 = np.array([1,2,3,4])
    # vector_2 = np.array([3,5,4,2])
    # cos_similarity = cos_sim(vector_1, vector_2)
    # print(cos_similarity)
    # 0.8198915917499229

    # Similarity between embeddings of the same word from different times
    # major_cos_similarity = get_cos_sim_different_models("major", model_t1, model_t2, cos_sim)
    # print(major_cos_similarity)
    # 0.19302374124526978

    # Average cosine similarity to neighborhood of words
    neighbors_old = ['brigadier', 'colonel', 'lieutenant', 'brevet', 'outrank']
    neighbors_new = ['significant', 'key', 'big', 'biggest', 'huge']
    print(get_average_cos_sim("major", neighbors_old, model_t1, cos_sim))
    # 0.6957747220993042
    print(get_average_cos_sim("major", neighbors_new, model_t1, cos_sim))
    # 0.27042335271835327
    print(get_average_cos_sim("major", neighbors_old, model_t2, cos_sim))
    # 0.2626224756240845
    print(get_average_cos_sim("major", neighbors_new, model_t2, cos_sim))
    # 0.6279034614562988

    ### The takeaway -- When comparing word embeddings from 40-60 year olds in the 1910s and 2000s,
    ###                 (i) cosine similarity to the neighborhood of words related to military ranks goes down;
    ###                 (ii) cosine similarity to the neighborhood of words related to significance goes up.


    """Latent Semantic Analysis"""

    import nltk
    nltk.download('brown')
    from nltk.corpus import brown
    documents = [brown.words(fileid) for fileid in brown.fileids()]

    # Exploring the corpus
    print("The news section of the Brown corpus contains {} documents.".format(len(documents)))
    for i in range(3):
        document = documents[i]
        print("Document {} has {} words: {}".format(i, len(document), document))
    # The news section of the Brown corpus contains 500 documents.
    # Document 0 has 2242 words: ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]
    # Document 1 has 2277 words: ['Austin', ',', 'Texas', '--', 'Committee', 'approval', ...]
    # Document 2 has 2275 words: ['Several', 'defendants', 'in', 'the', 'Summerdale', ...]

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stopwords_list = stopwords.words('english')

    # This will take a few minutes to run
    tfidf_matrix, vocabulary = create_tfidf_matrix(documents, stopwords_list)
    idf_values = get_idf_values(documents, stopwords_list)

    # print(tfidf_matrix.shape)
    # # (500, 40881)

    # print(tfidf_matrix[np.nonzero(tfidf_matrix)][:5])
    # # [5.96857651 2.1079054  3.         2.07572071 2.69897   ]

    # print(vocabulary[2000:2010])
    # # ['amoral', 'amorality', 'amorist', 'amorous', 'amorphous', 'amorphously', 'amortization', 'amortize', 'amory', 'amos']

    # print(calculate_sparsity(tfidf_matrix))
    # # 0.9845266994447298

    """SVD"""
    U, Sigma, VT = randomized_svd(tfidf_matrix, n_components=10, n_iter=100, random_state=42)

    salient_words = extract_salient_words(VT, vocabulary, 10)
    print(salient_words[1])
    # ['anode', 'space', 'theorem', 'v', 'q', 'c', 'p', 'operator', 'polynomial', 'af']

    print("We will fetch documents similar to document {} - {}...".format(3, ' '.join(documents[3][:50])))
    # We will fetch documents similar to document 3 - 
    # Oslo The most positive element to emerge from the Oslo meeting of North Atlantic Treaty Organization Foreign Ministers has been the freer , 
    # franker , and wider discussions , animated by much better mutual understanding than in past meetings . This has been a working session of an organization that...

    similar_doc_indices = get_similar_documents(U, Sigma, VT, 3, 5)
    for i in range(2):
        print("Document {} is similar to document 3 - {}...".format(similar_doc_indices[i], ' '.join(documents[similar_doc_indices[i]][:50])))
    # Document 61 is similar to document 3 - 
    # For a neutral Germany Soviets said to fear resurgence of German militarism to the editor of the New York Times : 
    # For the first time in history the entire world is dominated by two large , powerful nations armed with murderous nuclear weapons that make conventional warfare of the past...
    # Document 6 is similar to document 3 - 
    # Resentment welled up yesterday among Democratic district leaders and some county leaders at reports that Mayor Wagner had decided to seek a third term with Paul R. Screvane and Abraham D. Beame as running mates . 
    # At the same time reaction among anti-organization Democratic leaders and in the Liberal party... 
    
    query = ['Krim', 'attended', 'the', 'University', 'of', 'North', 'Carolina', 'to', 'follow', 'Thomas', 'Wolfe']
    print("We will fetch documents relevant to query - {}".format(' '.join(query)))
    relevant_doc_indices = document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, 5)

    for i in range(2):
        print("Document {} is relevant to query - {}...".format(relevant_doc_indices[i], ' '.join(documents[relevant_doc_indices[i]][:50])))
    # Document 90 is relevant to query - 
    # One hundred years ago there existed in England the Association for the Promotion of the Unity of Christendom . 
    # Representing as it did the efforts of only unauthorized individuals of the Roman and Anglican Churches , and urging a communion of prayer unacceptable to Rome , this association produced little...
    # Document 101 is relevant to query - To what extent and in what ways did Christianity affect the United States of America in the nineteenth century ? ? 
    # How far and in what fashion did it modify the new nation which was emerging in the midst of the forces shaping the revolutionary age ? ? To what...
