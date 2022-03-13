import sys, re, os, nltk, csv
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from textstat.textstat import textstat
from nltk.stem.wordnet import WordNetLemmatizer
import language_tool_python
import helpers

# 
dirname = os.path.dirname(__file__)

# 
transition_words = [('and', 'then'), ('besides'), ('equally', 'important'), ('finally'), ('further'),
                    ('furthermore'), ('nor'), ('next'), ('lastly'), ('what\'s', 'more'), ('moreover'), ('in', 'addition'),
                    ('first'), ('second'), ('third'), ('fourth'), ('whereas'), ('yet'), ('on', 'the', 'other', 'hand'), ('however'),
                    ('nevertheless'), ('on', 'the', 'contrary'), ('by', 'comparison'), ('compared', 'to'), ('up', 'against'),
                    ('balanced', 'against'), ('vis', 'a', 'vis'), ('although'), ('conversely'), ('meanwhile'), ('after', 'all'),
                    ('in', 'contrast'), ('although', 'this', 'may', 'be', 'true'), ('because'), ('since'), ('for', 'the', 'same', 'reason'),
                    ('obviously'), ('evidently'), ('indeed'), ('in', 'fact'), ('in', 'any', 'case'), ('that', 'is'), ('still'), ('in', 'spite', 'of'),
                    ('despite'), ('of', 'course'), ('once', 'in', 'a', 'while'), ('sometimes'), ('immediately'), ('thereafter'), ('soon'),
                    ('after', 'a', 'few', 'hours'), ('then'), ('later'), ('previously'), ('formerly'), ('in', 'brief'), ('as', 'I', 'have', 'said'),
                    ('as', 'I', 'have', 'noted'), ('as', 'has', 'been', 'noted'), ('definitely'), ('extremely'), ('obviously'), ('absolutely'),
                    ('positively'), ('naturally'), ('surprisingly'), ('always'), ('forever'), ('perennially'), ('eternally'), ('never'),
                    ('emphatically'), ('unquestionably'), ('without', 'a', 'doubt'), ('certainly'), ('undeniably'), ('without', 'reservation'),
                    ('following', 'this'), ('at', 'this', 'time'), ('now'), ('at', 'this', 'point'), ('afterward'), ('subsequently'), ('consequently'),
                    ('previously'), ('before', 'this'), ('simultaneously'), ('concurrently'), ('thus'), ('therefore'), ('hence'), ('for', 'example'),
                    ('for', 'instance'), ('in', 'this', 'case'), ('in', 'another', 'case'), ('on', 'this', 'occasion'), ('in', 'this', 'situation'),
                    ('take', 'the', 'case', 'of'), ('to', 'demonstrate'), ('to', 'illustrate'), ('as', 'an', 'illustration'), ('on', 'the', 'whole'),
                    ('summing', 'up'), ('to', 'conclude'), ('in', 'conclusion'), ('as', 'I', 'have', 'shown'), ('as', 'I', 'have', 'said'),
                    ('accordingly'), ('as', 'a', 'result')]
transitions_set = set(transition_words)
relevant_trigrams = [('IN', 'DT', 'NN'), ('VB', 'JJ', 'NNS'), ('VBZ', 'JJ', 'NNS'), ('PRP', 'TO', 'VB'),
                    ('VB', 'DT', 'NN'), ('DT', 'JJ', 'NNS'), ('CC', 'JJ', 'NN'), ('CC', 'PRP', 'VBZ'), ('.', 'NN', 'VBP'),
                    ('TO', 'VB', 'IN'), ('DT', 'NN', 'VBP'), ('DT', 'NNS', 'VBP'), ('PRP$', 'NN', 'CC'), ('NN', '.', 'WRB'),
                    ('JJ', 'NN', 'CC'), ('VBP', 'RB', 'JJ'), ('TO', 'VB', 'JJR'), ('VB', 'NN', 'IN'), ('VBN', 'TO', 'VB'),
                    ('JJ', 'IN', 'PRP'), ('NNS', '.', 'IN'), ('PRP', 'VBP', 'JJ'), ('IN', 'NN', '.'), ('RB', ',', 'NN'),
                    (',', 'DT', 'NNS'), ('NN', 'CC', 'TO'), ('NNS', 'RB', 'VBP'), ('JJ', 'NNS', ','), ('NN', '.', 'IN'),
                    (',', 'IN', 'NNS'), ('NN', 'IN', 'NNS'), ('VBZ', 'DT', 'JJ'), ('JJ', 'VBP', 'RB'), ('VBP', 'DT', 'NN'),
                    (',', 'PRP', 'RB'), ('JJ', 'NN', 'IN'), ('NNS', 'VBP', 'JJ'), ('VBZ', 'DT', 'NN'), ('MD', 'VB', 'PRP'),
                    ('DT', 'NNS', '.'), ('IN', 'PRP', 'VBZ'), ('NN', 'TO', 'VB'), ('VBZ', 'VBN', 'TO'), ('NN', '.', 'NNS'),
                    ('PRP', 'MD', 'VB'), ('PRP', 'VBD', 'DT'), ('IN', 'PRP', 'TO'), ('VB', 'IN', 'IN'), (',', 'IN', 'PRP'),
                    ('RB', 'VB', 'NNS'), ('VBP', 'RB', 'VB'), ('RB', 'VB', 'NN'), ('.', 'DT', 'NN'), ('DT', 'NN', 'VBZ'),
                    ('NN', 'IN', 'DT'), ('VBP', 'DT', 'JJ'), ('VBG', 'JJ', 'TO'), ('NNS', 'VBP', 'NN'), ('NNS', ',', 'NN'),
                    ('NNS', 'IN', 'NN'), ('NN', 'IN', 'NN'), ('VBP', 'JJR', 'NN'), ('VBD', 'TO', 'VB'), ('VB', 'JJ', 'VBZ'),
                    ('JJR', 'NN', 'CC'), ('NNS', '.', 'RB'), ('NNS', 'WDT', 'VBP'), ('VBG', 'PRP', 'TO'), ('NN', ',', 'JJ'),
                    ('VBP', 'JJ', 'NN'), ('NN', ',', 'CD'), ('IN', 'PRP', 'RB'), ('MD', 'VB', 'TO'), (',', 'PRP', 'MD'),
                    ('IN', 'CD', 'NNS'), (',', 'NN', 'VBP'), ('DT', 'NN', 'IN'), ('PRP', 'VBD', 'IN'), ('JJ', 'NN', 'MD'),
                    ('NN', 'IN', 'PRP$'), ('TO', 'NNS', 'MD'), ('NN', '.', 'DT'), ('NNS', 'JJ', 'IN'), ('NNS', 'IN', 'DT'),
                    ('.', 'DT', 'JJ'), ('PRP', 'NNS', ','), ('NNS', ',', 'EX'), ('IN', 'NN', ','), ('NN', 'MD', 'VB'),
                    ('PRP', 'RB', '.'), ('NNS', 'MD', 'VB'), ('JJ', '.', 'RB'), (',', 'PRP', 'VBD'), ('NNS', 'TO', 'VB'),
                    ('NN', 'VBZ', 'PRP'), ('NNS', 'IN', 'PRP'), ('VBD', 'DT', 'JJ'), ('WP', 'MD', 'VB'), ('IN', 'VBG', 'CC'),
                    ('IN', 'NN', 'IN'), ('JJ', ',', 'VBG'), ('MD', 'VB', 'NNS'), ('CC', 'WRB', 'PRP'), ('DT', 'NNS', 'IN'),
                    ('WRB', 'PRP', 'VBP'), ('DT', 'NNS', 'VBD'), ('RB', 'VB', 'IN'), ('NN', 'DT', 'NN'), ('DT', 'NN', '.'),
                    ('CC', 'VBG', 'IN'), ('VBP', 'JJR', 'NNS'), ('.', 'IN', 'IN'), ('IN', 'PRP$', 'NN'), ('VB', 'PRP$', 'NN'),
                    ('.', 'DT', 'MD'), ('RB', ',', 'PRP'), ('IN', 'DT', 'JJ'), ('.', 'IN', 'NN'), (',', 'PRP', 'VBP')]
relevant_trigram_set = set(relevant_trigrams)

# 
feature_categorys = [
    "identifier",
    "grammar", 
    "vocab", 
    "flow", 
    "ideas", 
    "coherence",
    "overall"
    ]

# 
category_feature_names = {
    "identifier":   [],
    "grammar":      [],
    "vocab":        [],
    "flow":         [],
    "ideas":        [],
    "coherence":    [],
    "overall":         []
}

# 
feature_category_row_list = {
    "identifier":   [],
    "grammar":      [],
    "vocab":        [] ,
    "flow":         [],
    "ideas":        [],
    "coherence":    [],
    "overall":          []
}

# 
feature_category_output_list = {
    "identifier":   [],
    "grammar":      [],
    "vocab":        [] ,
    "flow":         [],
    "ideas":        [],
    "coherence":    [],
    "overall":          []
}

# 
def download_nltk_resources():
    print("================================")
    print("download_nltk_resources: ")
    print("===")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    print("================================")

# 
def download_gammar_tool_resources():
    print("================================")
    print("download_gammar_tool_resources: ")
    print("===")
    gammar_tool = language_tool_python.LanguageTool('en-US')  # use a local server (automatically set up), language English
    print("================================")
    return gammar_tool

# will automatically process to the given category
# as well as the "overall"
def process_feature(
    category, 
    name, 
    value
    ):

    # 
    if(category == "all"):
        # 
        for feature in feature_categorys:
            if name not in category_feature_names[feature]:
                category_feature_names[feature].append(name)
        # 
        for feature in feature_categorys:
            feature_category_row_list[feature].append(value)
    elif(category == "overall"):
        # add to overall category
        if str(name) not in category_feature_names["overall"]:
            category_feature_names["overall"].append(name)
        feature_category_row_list["overall"].append(value)
    else:
        # add to feature specific category
        if str(name) not in category_feature_names[category]:
            category_feature_names[category].append(name)
        feature_category_row_list[category].append(value)
        # add to overall category
        if str(name) not in category_feature_names["overall"]:
            category_feature_names["overall"].append(name)
        feature_category_row_list["overall"].append(value)

# 
def import_headword_collection():

    # 
    total_headwords_lists = 29
    vocab_lists = []

    # 
    print("================================")
    print("import_headword_collection: ")
    print("===")

    # 
    for x in range(1, total_headwords_lists+1):
        vocab_list = []

        a_file = open(
            os.path.join(
                dirname, 
                str(
                    "headwords&basewords/basewrd"+
                    str(x)+
                    ".txt"
                    )),
            "r",  
            encoding='latin-1'
            )

        for line in a_file:
            vocab_list.append(line.lower().strip())
        a_file.close()

        vocab_lists.append(vocab_list)

    # 
    print("vocab_lists[0][0]: " + vocab_lists[0][0])
    print("================================")

    # 
    return vocab_lists

# 
def import_essays(
    dataset,
    type, 
    version,
    cols,
    rows
    ):
 
    # 
    input_file_name = helpers.structured_essays_file_name(
        dataset,
        type, 
        version,
        rows
        )
    
    # 
    print("================================")
    print("import_essays: ")
    print("===")
    print(" dataset: " + dataset)
    print(" type: " + type)
    print(" version: " + version)
    print(" cols: " + cols)
    print(" rows: " + rows)
    print(" input_file_name: " + input_file_name)

    # 
    cols = int(cols)
    rows = int(rows)
    data = [["" for i in range(cols)] for j in range(rows)]

    # 
    r_count = -1
    with open(input_file_name) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            if (row[0].isdigit()):   

                # 
                r_count += 1
                data[r_count][0] = row[0] # id
                data[r_count][1] = row[1] # essay

                # 
                if(type == "overall"):
                    data[r_count][2] = row[2] # overall
                elif(type == "categorized"):

                    if(version == "1"):
                        data[r_count][2] = row[2] # ideas
                        data[r_count][3] = row[3] # flow
                        data[r_count][4] = row[4] # coherence
                        data[r_count][5] = row[5] # vocab
                        data[r_count][6] = row[6] # grammar
                        data[r_count][7] = row[7] # overall

                    if(version == "2"):
                        data[r_count][2] = row[2] # "Writing_Applications"
                        data[r_count][3] = row[3] # "Language_Conventions"
                        
                    elif(version == "8"):
                        data[r_count][2] = row[2] # overall
                        data[r_count][3] = row[3] # Ideas_and_Content
                        data[r_count][4] = row[4] # Organization
                        data[r_count][5] = row[5] # Voice
                        data[r_count][6] = row[6] # Word_Choice
                        data[r_count][7] = row[7] # Sentence_Fluency
                        data[r_count][8] = row[8] # Conventions

                    elif(version == "7"):
                        data[r_count][2] = row[2] # overall
                        data[r_count][3] = row[3] # Ideas
                        data[r_count][4] = row[4] # Organization
                        data[r_count][5] = row[5] # Style
                        data[r_count][6] = row[6] # Conventions

    # to test/ view data structure
    # for row in data:
    #     print(row)

    # 
    print(" data[0][0]: " + data[0][0])
    print(" data[0][1][0:10]: " + data[0][1][0:10])
    print(" data[0][2]: " + data[0][2])
    print("================================")

    # 
    return data

# 
def generate_features(
    essays,
    headword_collection,
    type
    ):

    # 
    download_nltk_resources()
    gammar_tool = download_gammar_tool_resources()

    # 
    print("================================")
    print("generate_features: ")
    print("===")
    print(" essays[0][0]: " + essays[0][0])
    print("===")
    print(" | " + "id" + " | " + "score" + " | " + "essay[0:30]") 
    print("================================")


    # 
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords = nltk.corpus.stopwords.words('english')
    lmtzr = WordNetLemmatizer()
    one_iteration = False

    #   
    for row in essays:
       
        # 
        id = row[0]
        essay = row[1]
        process_feature("all", "id", id)
        if(type == "overall"):
            process_feature("all", "overall_score", row[2])                  
        elif(type == "categorized"):
                if(version == "1"):
                    process_feature("all", "overall_score", row[7])            
                    process_feature("all", "ideas_score", row[2])            
                    process_feature("all", "flow_score", row[3])             
                    process_feature("all", "coherence_score", row[4])       
                    process_feature("all", "vocab_score", row[5])          
                    process_feature("all", "grammar_score", row[6])           
                if(version == "2"):
                    process_feature("all", "Writing_Applications", row[2])     
                    process_feature("all", "Language_Conventions", row[3])     
                elif(version == "8"):
                    process_feature("all", "overall_score", row[2])           
                    process_feature("all", "Ideas_and_Content", row[3])        
                    process_feature("all", "Organization", row[4])          
                    process_feature("all", "Voice", row[5])                      
                    process_feature("all", "Word_Choice", row[6])              
                    process_feature("all", "Sentence_Fluency", row[7])         
                    process_feature("all", "Conventions", row[8])             
                elif(version == "7"):
                    process_feature("all", "overall_score", row[2])         
                    process_feature("all", "Ideas", row[3])                
                    process_feature("all", "Organization", row[4])      
                    process_feature("all", "Style", row[5])               
                    process_feature("all", "Conventions", row[6])        
        print(" | " + id + " | " + essay[0:30]) 

        # 
        if(id == ""):
            print("blank row")
            continue

        # <><><><><><><><><><><><>
        #     TOKENISING
        # <><><><><><><><><><><><>

        sentences = sent_tokenize(essay)
        words = word_tokenize(essay)

        # with punctuation, not lowered
        tokens = word_tokenize(essay)
        tagged = nltk.pos_tag(tokens)
        num_sents = len(sent_tokenize(essay))

        # with punctuation, lowered
        essay_low = essay.strip().lower()
        tokens_low = word_tokenize(essay_low)
        tagged_low = nltk.pos_tag(tokens_low)

        # without punctuation, not lowered 
        tokens_np = tokenizer.tokenize(essay)
        num_tokens = len(tokens_np)

        # without punctuation, lowered
        tokens_low_np = tokenizer.tokenize(essay_low)
        types = set(tokens_low_np)
        num_types = len(types)

        # content and function words
        content_tokens = [w for w in tokens_np if w not in stopwords]
        content_types = [w for w in types if w not in stopwords]
        function_tokens = [w for w in tokens_np if w in stopwords]
        function_types = [w for w in types if w in stopwords]

        # <><><><><><><><><><><><>
        #     FEATURES: MISC
        # <><><><><><><><><><><><>

        # total words
        total_words = len(words)
        process_feature("all", "total_words", total_words)

        # total sentences
        total_sentences = len(sentences)
        process_feature("all", "total_sentences", total_sentences)

        # total words per sentence
        words_div_sentences = total_words/total_sentences
        process_feature("all", "words_div_sentences", words_div_sentences)

        # sentence density
        sent_density = round(num_sents / num_tokens * 100, 2)
        process_feature("all", "sent_density", sent_density)

        # <><><><><><><><><><><><>
        #     FEATURES: GRAMMAR
        # <><><><><><><><><><><><>

        # percent of transition word types in essay
        found_transitions = transitions_set & types 
        pct_transitions = round(len(found_transitions) / len(transitions_set), 4)
        process_feature("grammar", "pct_transitions", pct_transitions)

        # percent of trigrams in essay
        a, b = zip(*tagged)
        trigram_set = set(nltk.trigrams(b))
        found_trigrams = relevant_trigram_set & trigram_set
        pct_rel_trigrams = round(len(found_trigrams) / len(relevant_trigram_set) * 100, 2)
        process_feature("grammar", "pct_rel_trigrams", pct_rel_trigrams)

        # simple "language_tool_python" lib checks
        grammar_errors_ltp_list = gammar_tool.check(essay)
        grammar_errors_ltp_by_words = len(grammar_errors_ltp_list)/total_words
        process_feature("grammar", "grammar_errors_ltp_by_words", grammar_errors_ltp_by_words)

        # <><><><><><><><><><><><>
        #     FEATURES: VOCAB
        # <><><><><><><><><><><><>

        # textstat: difficult_words
        ts_difficult_words = textstat.difficult_words(essay)
        process_feature("vocab", "difficult_words", ts_difficult_words)

        # BNC/COCA : headwords & basewords:
        for headwords in headword_collection:
            matches = len(set(words).intersection(headwords))
            proportion_matches_total_words = matches/total_words
            proportion_matches_headwords = matches/len(headwords)
            process_feature("vocab", "proportion_vocab_list["+headwords[0]+"]_total_words", proportion_matches_total_words)
            process_feature("vocab", "proportion_vocab_list["+headwords[0]+"]_headwords", proportion_matches_headwords)
        
        # unique words
        ttr = round(num_types / num_tokens * 100, 2)
        ttrr = round(num_types / num_tokens, 2)
        process_feature("vocab", "unique_words", ttr)
        process_feature("vocab", "ttrr", ttrr)
        process_feature("vocab", "num_types", num_types)

        # <><><><><><><><><><><><>
        #     FEATURES: FLOW
        # <><><><><><><><><><><><>

        # Formulas
        flesch_reading_ease             = textstat.flesch_reading_ease(essay)
        flesch_kincaid_grade            = textstat.flesch_kincaid_grade(essay)
        smog_index                      = textstat.smog_index(essay)
        automated_readability_index     = textstat.automated_readability_index(essay)
        coleman_liau_index              = textstat.coleman_liau_index(essay)
        linsear_write_formula           = textstat.linsear_write_formula(essay)
        dale_chall_readability_score    = textstat.dale_chall_readability_score(essay)
        text_standard                   = textstat.text_standard(essay, float_output=True)
        spache_readability              = textstat.spache_readability(essay)
        reading_time                    = textstat.reading_time(essay, ms_per_char=14.69)
        crawford                        = textstat.crawford(essay)
        wiener_sachtextformel           = textstat.wiener_sachtextformel(essay, 1)

        # Aggregates and Averages
        monosyllabcount                 = textstat.monosyllabcount(essay)
        polysyllabcount                 = textstat.polysyllabcount(essay)
        letter_count                    = textstat.letter_count(essay, ignore_spaces=True)
        char_count                      = textstat.char_count(essay, ignore_spaces=True)
        sentence_count                  = textstat.sentence_count(essay)
        lexicon_count                   = textstat.lexicon_count(essay, removepunct=True)
        syllable_count                  = textstat.syllable_count(essay)
        ts_avg_len_sent                 = textstat.avg_sentence_length(essay)
        ts_avg_sent_per_word            = textstat.avg_sentence_per_word(essay)
        ts_avg_syllab_per_word          = textstat.avg_syllables_per_word(essay)

        #
        process_feature("flow", "flesch_reading_ease", flesch_reading_ease)
        process_feature("flow", "flesch_kincaid_grade", flesch_kincaid_grade)
        process_feature("flow", "smog_index", smog_index)
        process_feature("flow", "automated_readability_index", automated_readability_index)
        process_feature("flow", "coleman_liau_index", coleman_liau_index) 
        process_feature("flow", "linsear_write_formula", linsear_write_formula) 
        process_feature("flow", "dale_chall_readability_score", dale_chall_readability_score)
        process_feature("flow", "text_standard", text_standard)
        process_feature("flow", "spache_readability", spache_readability)
        process_feature("flow", "reading_time", reading_time)
        process_feature("flow", "crawford", crawford)
        process_feature("flow", "wiener_sachtextformel", wiener_sachtextformel)
        
        #
        process_feature("flow", "monosyllabcount", monosyllabcount)
        process_feature("flow", "polysyllabcount", polysyllabcount)
        process_feature("flow", "letter_count", letter_count)
        process_feature("flow", "char_count", char_count)
        process_feature("flow", "sentence_count", sentence_count)
        process_feature("flow", "lexicon_count", lexicon_count) 
        process_feature("flow", "syllable_count", syllable_count) 
        process_feature("flow", "ts_avg_len_sent", ts_avg_len_sent)
        process_feature("flow", "ts_avg_sent_per_word", ts_avg_sent_per_word)
        process_feature("flow", "ts_avg_syllab_per_word", ts_avg_syllab_per_word)

        # <><><><><><><><><><><><>
        #     FEATURES: COHERENCE
        # <><><><><><><><><><><><>

        # X-grams on Lemmas:
        # for all unique words
        lemma_types_list = []
        for word in types: # types = unique words (lowercase)
            lemma_types = lmtzr.lemmatize(word)
            # add all unique words converted down to thier respected dictionary equiv.
            lemma_types_list.append(lemma_types) 
            # could be outside loop as is doing operation on the list we're creating...
            bigram_lemma_types = nltk.bigrams(lemma_types_list)
            trigram_lemma_types = nltk.trigrams(lemma_types_list)
        nlemma_types = len(lemma_types_list)                    # number of unique lemmas
        n_bigram_lemma_types = len(list(bigram_lemma_types))    # number of unique bigram-lemmas
        n_trigram_lemma_types = len(list(trigram_lemma_types))  # number of unique trigram-lemmas
        process_feature("coherence", "nlemma_types", nlemma_types)
        process_feature("coherence", "n_bigram_lemma_types", n_bigram_lemma_types)
        process_feature("coherence", "n_trigram_lemma_types", n_trigram_lemma_types)
        # for all non-unique words
        lemma_tokens_list = []
        for word in tokens_np:
            lemma_tokens = lmtzr.lemmatize(word)
            lemma_tokens_list.append(lemma_tokens)
            bigram_lemmas = nltk.ngrams(lemma_tokens_list,2)
            trigram_lemmas = nltk.ngrams(lemma_tokens_list,3)
        nlemmas = len(lemma_tokens_list)
        n_bigram_lemmas = len(list(bigram_lemmas))
        n_trigram_lemmas = len(list(trigram_lemmas))
        process_feature("coherence", "nlemmas", nlemmas)
        process_feature("coherence", "n_bigram_lemmas", n_bigram_lemmas)
        process_feature("coherence", "n_trigram_lemmas", n_trigram_lemmas)

        # content words:
        ncontent_tokens = len(content_tokens)
        ncontent_types = len(content_types)
        process_feature("coherence", "ncontent_types", ncontent_types)
        process_feature("coherence", "ncontent_tokens", ncontent_tokens)
        #
        content_ttr = 0
        try:
            content_ttr = round(ncontent_types/ncontent_tokens,4)
        except ZeroDivisionError:
            content_ttr = 0
        process_feature("coherence", "content_ttr", content_ttr)

        # function words:
        nfunction_tokens = len(function_tokens)
        nfunction_types = len(function_types)
        process_feature("coherence", "nfunction_tokens", nfunction_tokens)
        process_feature("coherence", "nfunction_types", nfunction_types)
        try:
            function_ttr = round(nfunction_types/nfunction_tokens,4)
        except ZeroDivisionError:
            function_ttr = 1
        process_feature("coherence", "function_ttr", function_ttr)

        # nouns:
        nouns = []
        ttr_nouns = 0
        for word, tag in tagged:
            if re.search(r'\b(NN(S|P|PS))\b', tag):
                nouns.append(word)
        try:
            ttr_nouns = round(len(set(nouns))/len(nouns),4)
        except ZeroDivisionError:
            ttr_nouns = 0
        nouns_total = len(nouns)
        nouns_unique = len(set(nouns))
        process_feature("coherence", "nouns_total", nouns_total)
        process_feature("coherence", "nouns_unique", nouns_unique)
        process_feature("coherence", "ttr_nouns", ttr_nouns)

        # determiners
        determiners = re.findall(r'\b(DT)\b', str(tagged), flags=re.I)
        determiners_unique = len(set(determiners))
        determiners_total = len(determiners)
        determiners_per_words = round(determiners_total/len(tokens_np), 5)
        process_feature("coherence", "determiners_unique", determiners_unique)
        process_feature("coherence", "determiners_per_words", determiners_per_words)
        process_feature("coherence", "determiners_total", determiners_total)

        # conjunctions
        conjunctions = re.findall(r'\b(and|but)\W+(CC)\b', str(tagged), flags=re.I)
        conjunctions_unique = len(set(conjunctions))
        conjunctions_total = len(conjunctions)
        conjunctions_per_words = round(conjunctions_total/len(tokens_np), 5)
        process_feature("coherence", "conjunctions_unique", conjunctions_unique)
        process_feature("coherence", "conjunctions_total", conjunctions_total)
        process_feature("coherence", "conjunctions_per_words", conjunctions_per_words)

        # pronouns
        pronouns = re.findall(r'\b(he|she|it|his|hers|him|her|they|them|their)\b', str(words), flags=re.I)
        pronouns_unique = len(set(conjunctions))
        pronouns_total = len(pronouns)
        pronouns_density = round(pronouns_total/len(tokens_np), 5)
        pronouns_noun_ratio = 0
        try:
            pronouns_noun_ratio = round(pronouns_total/len(nouns), 2)
        except ZeroDivisionError:
            pronouns_noun_ratio = 0
        process_feature("coherence", "pronouns_unique", pronouns_unique)
        process_feature("coherence", "pronouns_total", pronouns_total)
        process_feature("coherence", "pronouns_density", pronouns_density)
        process_feature("coherence", "pronouns_noun_ratio", pronouns_noun_ratio)

        # <><><><><><><><><><><><>
        #     FEATURES: IDEAS
        # <><><><><><><><><><><><>

        # 

        # <><><><><><><><><><><><>
        #     OUTPUT
        # <><><><><><><><><><><><>

        # if the output lists are empty, then it is the first
        # iteration, if so, then add each of the created colum names
        # as the first row to the document
        # if one of them is empty, then they're all empty (in thoery)
        if not feature_category_output_list['identifier']:
            for feature in feature_categorys:
                feature_category_output_list[feature].append(
                    category_feature_names[feature]
                    )

        # append feature values to output list
        for feature in feature_categorys:
            feature_category_output_list[feature].append(
                feature_category_row_list[feature]
                )

        # <><><><><><><><><><><><>

        # clear  the row
        for feature in feature_categorys:
            feature_category_row_list[feature] = []

        # <><><><><><><><><><><><>

        if one_iteration:
            break

    # <><><><><><><><><><><><>

    print("================================")

# 
def export_features(
    dataset,
    type,
    version,
    rows
    ):

    # 
    print("================================")
    print("export_features: ")
    print("===")
    print(" output_file_name(s): ")

    # 
    categorys = []
    if(type == "categorized"):
        categorys.extend(feature_categorys)
        categorys.remove("identifier")
        # categorys.remove("overall")

    elif(type == "overall"):
        categorys = [
            "overall"
            ]

    # 
    for feature in categorys:

        # 
        output_file_name = helpers.feature_category_output_file_name(
            dataset,
            type, 
            version,
            rows,
            feature,
            )
        print("  " + output_file_name)

        # 
        with open(
            output_file_name, 
            'w+', 
            encoding='utf-8'
            ) as output_file:

            # ... print each record into file
            for line in feature_category_output_list[feature]:
                print(*line, sep='\t', file=output_file)
    print("================================")

# 
def run(
    dataset,
    type, 
    version,
    rows,
    cols
    ):

    # 
    print("================================")
    print("main : ")
    print("===")
    print(" dataset : " + dataset)
    print(" type: " + type)
    print(" version : " + version)
    print(" rows: " + rows)
    print(" cols: " + cols)
    print("================================")
    
    # 
    if(helpers.params_ok(type)): pass
    else: return

    # 
    headword_collection = import_headword_collection()

    # 
    essays = import_essays(
        dataset,
        type, 
        version,
        cols,
        rows
        )

    # 
    generate_features(essays,headword_collection, type)

    #
    export_features(
        dataset,
        type,
        version,
        rows
        ) 

# ================================
#   run: bash level
# ================================ 

if __name__ == "__main__":
    dataset = str(sys.argv[1])
    type = str(sys.argv[2])
    version = str(sys.argv[3])
    rows = str(sys.argv[4])
    cols = str(sys.argv[5])

    run(dataset, type, version, rows, cols)

# ================================
#   run: python level
# ================================ 

# python3 -u "feature_extractor.py"

# 
# dataset = "vuw"         # vuw, hewlett
# type = "overall"        # categorized
# version = "1"           # or sub type
# rows = "113"            # total id's in file
# cols = "3"              # 3, 8

# # 
# run(dataset,type, version, rows, cols)