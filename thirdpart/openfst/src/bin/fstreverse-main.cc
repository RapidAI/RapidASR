// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Reverses the paths in an FST.

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/script/fst-class.h>
#include <fst/script/reverse.h>

DECLARE_bool(require_superinitial);

int fstreverse_main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;
  using fst::script::VectorFstClass;

  string usage = "Reverses the paths in an FST.\n\n  Usage: ";
  usage += argv[0];
  usage += " [in.fst [out.fst]]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  const string in_name =
      (argc > 1 && (strcmp(argv[1], "-") != 0)) ? argv[1] : "";
  const string out_name = argc > 2 ? argv[2] : "";

  std::unique_ptr<FstClass> ifst(FstClass::Read(in_name));
  if (!ifst) return 1;

  VectorFstClass ofst(ifst->ArcType());

  s::Reverse(*ifst, &ofst, FLAGS_require_superinitial);

  return !ofst.Write(out_name);
}
