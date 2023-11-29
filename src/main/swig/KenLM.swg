%module KenLM
%{
#include <iostream>
#include <string>
#include "lm/word_index.hh"
#include "lm/return.hh"
#include "lm/state.hh"
#include "lm/virtual_interface.hh"
#include "util/mmap.hh"
#include "lm/config.hh"
#include "lm/model.hh"
#include "lm/binary_format.hh"
#include "util/exception.hh"
%}

%include "std_string.i"

SWIG_JAVABODY_PROXY(public, public, SWIGTYPE)
SWIG_JAVABODY_TYPEWRAPPER(public, public, public, SWIGTYPE)

typedef unsigned long long uint64_t;
typedef unsigned int size_t;

namespace lm {
    typedef unsigned int WordIndex;

    struct FullScoreReturn {
        float prob;
        unsigned char ngram_length;
    };
};

%nodefaultctor lm::base::Vocabulary;
%nodefaultctor lm::base::Model;


%{

class Evaluator
{
private:
    std::vector<char> ctx1;
    std::vector<char> ctx2;
    const lm::base::Model *model;
    std::vector<float> probs;
    size_t n_tokens;
    size_t n_processed;

    double absorb_prob(float value)
    {
        probs.push_back(value);
        size_t sz = probs.size();
        if (sz == probs.capacity())
        {
            size_t half = sz / 2;
            std::nth_element(probs.begin(), probs.begin() + half, probs.end(), std::less<>());
            double result = 0;
            for (size_t i = half; i < sz; ++i)
            {
                result += probs[i];
            }
            n_processed += sz - half;
            probs.erase(probs.begin() + half, probs.end());
            return result;
        }
        return 0;
    }

    double finish_prob(size_t position)
    {
        size_t sz = probs.size();
        size_t half = probs.capacity() / 2;
        size_t num_ignored = std::min(std::min(position, half), sz);
        std::nth_element(probs.begin(), probs.begin() + num_ignored, probs.end(), std::less<>());
        double result = 0;
        for (size_t i = num_ignored; i < sz; ++i)
        {
            result += probs[i];
        }
        n_processed += sz - num_ignored;
        return result;
    }

public:
    Evaluator(const lm::base::Model *model, size_t prob_size) : model(model), probs(prob_size * 2, 0.0f)
    {
        size_t state_sz = model->StateSize();
        ctx1.resize(state_sz);
        ctx2.resize(state_sz);
    }

    double evaluateSum(unsigned char *data_raw, size_t sz)
    {
        const size_t npos = StringPiece::npos;
        const lm::base::Model &model = *this->model;
        const lm::base::Vocabulary &vocabulary = model.BaseVocabulary();
        const char *zero_data = static_cast<const char *>(model.BeginSentenceMemory());
        ctx1.assign(zero_data, zero_data + model.StateSize());
        void *ctx_cur = ctx1.data();
        void *ctx_next = ctx2.data();
        probs.clear();
        n_processed = 0;

        StringPiece data(reinterpret_cast<const char*>(data_raw), sz);
        size_t tokens = 0;
        double score = 0.0;
        while (!data.empty())
        {
            size_t offset = data.find(' ');
            StringPiece word;
            if (offset == npos)
            {
                word = data;
                data.clear();
            }
            else
            {
                word = data.substr(0, offset);
                data = data.substr(offset + 1);
            }
            lm::WordIndex idx = vocabulary.Index(word);
            lm::FullScoreReturn local_score = model.BaseFullScore(ctx_cur, idx, ctx_next);
            score += local_score.prob;
            tokens += 1;
            std::swap(ctx_cur, ctx_next);
        }
        lm::FullScoreReturn eos_score = model.BaseFullScore(ctx_cur, vocabulary.EndSentence(), ctx_next);
        score += eos_score.prob;
        n_tokens = tokens;
        return score;
    }

    double evaluateNoOutliers(unsigned char *data_raw, size_t sz, float outliers)
    {
        if (outliers == 0)
        {
            return evaluateSum(data_raw, sz);
        }

        if (outliers < 0 || outliers > 1)
        {
            return 0;
        }

        const size_t npos = StringPiece::npos;
        const lm::base::Model &model = *this->model;
        const lm::base::Vocabulary &vocabulary = model.BaseVocabulary();
        const char *zero_data = static_cast<const char *>(model.BeginSentenceMemory());
        ctx1.assign(zero_data, zero_data + model.StateSize());
        void *ctx_cur = ctx1.data();
        void *ctx_next = ctx2.data();
        probs.clear();
        n_processed = 0;

        StringPiece data(reinterpret_cast<const char*>(data_raw), sz);
        size_t tokens = 0;
        double score = 0.0;
        while (!data.empty())
        {
            size_t offset = data.find(' ');
            StringPiece word;
            if (offset == npos)
            {
                word = data;
                data.clear();
            }
            else
            {
                word = data.substr(0, offset);
                data = data.substr(offset + 1);
            }
            lm::WordIndex idx = vocabulary.Index(word);
            lm::FullScoreReturn local_score = model.BaseFullScore(ctx_cur, idx, ctx_next);
            score += absorb_prob(local_score.prob);
            tokens += 1;
            std::swap(ctx_cur, ctx_next);
        }
        lm::FullScoreReturn eos_score = model.BaseFullScore(ctx_cur, vocabulary.EndSentence(), ctx_next);
        score += absorb_prob(eos_score.prob);
        n_tokens = tokens;
        return score + finish_prob(size_t(outliers * tokens));
    }

    size_t numNonOutlierTokens() const { return this->n_processed; }
};
%}

%include "java/various.i"

namespace lm::base {
    class Vocabulary {
    public:
        lm::WordIndex BeginSentence() const;
        lm::WordIndex EndSentence() const;
        lm::WordIndex NotFound() const;
        lm::WordIndex Index(const char *str) const;
    };

    class Model {
    public:
        void BeginSentenceWrite(void *to) const;
        void NullContextWrite(void *to) const;
        unsigned int Order() const;
        const lm::base::Vocabulary &BaseVocabulary() const;
        float BaseScore(const void *in_state, const lm::WordIndex new_word, void *out_state) const;
        lm::FullScoreReturn BaseFullScore(const void *in_state, const lm::WordIndex new_word, void *out_state) const;
    };
};

class Evaluator {
public:
    Evaluator(const lm::base::Model* model, size_t buffer);

    %apply unsigned char *NIOBUFFER { unsigned char *data_raw };
    double evaluateNoOutliers(unsigned char *data_raw, size_t sz, float outliers);

    %apply unsigned char *NIOBUFFER { unsigned char *data_raw };
    double evaluateSum(unsigned char *data_raw, size_t sz);

    size_t numNonOutlierTokens() const;
};


namespace util {
    enum LoadMethod {
        LAZY,
        POPULATE_OR_LAZY,
        POPULATE_OR_READ,
        READ,
        PARALLEL_READ,
    };
};

namespace lm::ngram {
    class State {
    public:
        int Compare(const lm::ngram::State &other) const;
    };

    uint64_t hash_value(const lm::ngram::State &state);

    class Config {
    public:
        bool show_progress;
        float unknown_missing_logprob;
        float probing_multiplier;
        size_t  building_memory;
        std::ostream *messages;
        std::string temporary_directory_prefix;
        util::LoadMethod load_method;
    };

    %javaexception("com.github.jbaiter.kenlm.ModelException") LoadVirtual {
        try {
            $action
        } catch (lm::ConfigException &e) {
            jclass clazz = jenv->FindClass("com/github/jbaiter/kenlm/ConfigException");
            jenv->ThrowNew(clazz, e.what());
            return $null;
        } catch (lm::FormatLoadException &e) {
            jclass clazz = jenv->FindClass("com/github/jbaiter/kenlm/FormatLoadException");
            jenv->ThrowNew(clazz, e.what());
            return $null;
        } catch (lm::VocabLoadException &e) {
            jclass clazz = jenv->FindClass("com/github/jbaiter/kenlm/VocabLoadException");
            jenv->ThrowNew(clazz, e.what());
            return $null;
        } catch (lm::LoadException &e) {
            jclass clazz = jenv->FindClass("com/github/jbaiter/kenlm/LoadException");
            jenv->ThrowNew(clazz, e.what());
            return $null;
        } catch (util::Exception &e) {
            jclass clazz = jenv->FindClass("com/github/jbaiter/kenlm/ModelException");
            jenv->ThrowNew(clazz, e.what());
            return $null;
        }
    }
    %newobject LoadVirtual;
    lm::base::Model* LoadVirtual(const char *file_name,
                                 const lm::ngram::Config &config = lm::ngram::Config());
};
