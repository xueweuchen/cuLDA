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
  args.AddOption<double>("alpha", 0.1);
  args.AddOption<double>("beta", 0.1);
  args.AddOption<int>("topic", 50);
  args.AddOption<std::string>("file_name", "");
  args.AddOption<std::string>("stopwords", "");
  args.AddOption<std::string>("model_path", "model.dat");
  args.AddOption<std::string>("output", "output.dat");
  args.AddOption<std::string>("phase", "train");
  args.AddOption<int>("max_iter", 100);
  args.AddOption<std::string>("type", "AliasLDA");
  args.Parse(argc, argv);

  std::string file_name = args.GetOption<std::string>("file_name");
  std::string stopwords_ = args.GetOption<std::string>("stopwords");
  std::string model_path = args.GetOption<std::string>("model_path");
  std::string output = args.GetOption<std::string>("output");
  std::string phase = args.GetOption<std::string>("phase");
  std::string type = args.GetOption<std::string>("type");
  int topic = args.GetOption<int>("topic");
  int max_iter = args.GetOption<int>("max_iter");
  double alpha = args.GetOption<double>("alpha");
  double beta = args.GetOption<double>("beta");
  

  if (phase == "train") {
    Docs docs(file_name, stopwords_);
    auto lda = Registry::Create(type);
    lda->Init(docs, topic, alpha, beta);
    lda->Estimate(max_iter);
    lda->SaveModel(model_path);
    lda->Release();
  } else if (phase == "test") {
    Docs docs(file_name, stopwords_);
    auto lda = Registry::Create(type);
    lda->LoadModel(model_path);
    std::ofstream fout(output);
    for (auto test_doc : docs.GetDoclist()) {
      lda->InferInit(test_doc);
      std::vector<double> topics;
      lda->Inference(max_iter, topics);
      util::DumpVector(topics, fout);
    }
    lda->Release();
  } else {
    std::cout << "Unknown phase!" << std::endl;
  }
  return 0;
}