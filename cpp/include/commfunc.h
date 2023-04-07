#pragma once 

inline int getInputName(Ort::Session* session, string& inputName,int nIndex=0) {
    size_t numInputNodes = session->GetInputCount();
    if (nIndex == -1)
        return numInputNodes;
    if (numInputNodes > nIndex) {
        Ort::AllocatorWithDefaultOptions allocator;
        {
            char* t = session->GetInputName(nIndex, allocator);
            inputName = t;
            allocator.Free(t);
        }
    }
}

inline int getInputNameAll(Ort::Session* session, vector<string>&InputNames)
{
    size_t numInputNodes = session->GetInputCount();
   for(int i=0; i< numInputNodes; i++)
   {
        Ort::AllocatorWithDefaultOptions allocator;
        {
            char* t = session->GetInputName(i, allocator);
            InputNames.push_back( t);
            allocator.Free(t);
        }
   }

   return numInputNodes;

}


inline int  getOutputName(Ort::Session* session, string& outputName, int nIndex = 0) {
    size_t numOutputNodes = session->GetOutputCount();

    if (nIndex == -1)
        return numOutputNodes;
    if (numOutputNodes > nIndex) {
        Ort::AllocatorWithDefaultOptions allocator;
        {
            char* t = session->GetOutputName(nIndex, allocator);
            outputName = t;
            allocator.Free(t);
        }
    }

    return numOutputNodes;
}


inline int getOutputNameAll(Ort::Session* session, vector<string>& OutputNames) {
    size_t numOutputNodes = session->GetOutputCount();

    for (int i = 0; i < numOutputNodes; i++)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        {
            char* t = session->GetOutputName(i, allocator);
            OutputNames.push_back(t);
            allocator.Free(t);
        }
    }
    return numOutputNodes;
}