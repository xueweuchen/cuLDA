#include "util.h"
#include "documents.h"
#include "cpulda.h"
#include "gpulda.h"
#include "args.h"
#include "factory.h"

REGISTER_CLASS("AliasLDA", AliasLDA);
REGISTER_CLASS("MMLDA", MMLDA);
REGISTER_CLASS("GibbsLDA", GibbsLDA);
REGISTER_CLASS("LightLDA", LightLDA);


int main(int argc, char *argv[]) {
  util::ArgParse args;
  args.add_option<double>("alpha", 0.1);
  args.add_option<double>("beta", 0.1);
  args.add_option<int>("topic", 50);
  args.add_option<std::string>("file_name", "");
  args.add_option<std::string>("stopwords", "");
  args.add_option<std::string>("model_path", "model.dat");
  args.add_option<std::string>("output", "output.dat");
  args.add_option<std::string>("phase", "train");
  args.add_option<int>("max_iter", 100);
  args.add_option<std::string>("type", "AliasLDA");
  args.parse(argc, argv);

  std::string file_name = args.get_option<std::string>("file_name");
  std::string stopwords = args.get_option<std::string>("stopwords");
  std::string model_path = args.get_option<std::string>("model_path");
  std::string output = args.get_option<std::string>("output");
  std::string phase = args.get_option<std::string>("phase");
  std::string type = args.get_option<std::string>("type");
  int topic = args.get_option<int>("topic");
  int max_iter = args.get_option<int>("max_iter");
  double alpha = args.get_option<double>("alpha");
  double beta = args.get_option<double>("beta");
  

  if (phase == "train") {
    Docs docs(file_name, stopwords);
    auto lda = Registry::create(type);
    lda->init(docs, topic, alpha, beta);
    lda->estimate(max_iter);
    lda->save_model(model_path);
    lda->release();
  } else if (phase == "test") {
    Docs docs(file_name, stopwords);
    auto lda = Registry::create(type);
    lda->load_model(model_path);
    std::ofstream fout(output);
    for (auto test_doc : docs.get_doclist()) {
      lda->infer_init(test_doc);
      std::vector<double> topics;
      lda->inference(max_iter, topics);
      util::dump_vector(topics, fout);
    }
    lda->release();
  } else {
    std::cout << "Unknown phase!" << std::endl;
  }
  return 0;
}