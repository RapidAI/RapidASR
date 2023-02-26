#include "precomp.h"

Model *create_model(const char *path)
{
    Model *mm;


    mm = new paraformer::ModelImp(path);

    return mm;
}
