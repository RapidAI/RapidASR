#include "precomp.h"

Model *create_model(const char *path, int mode)
{
    Model *mm;


    mm = new paraformer::ModelImp(path, mode);

    return mm;
}
