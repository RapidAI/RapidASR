#include <stdlib.h>
#include <stdio.h>
#include "librpasrapi.h"
#define TEST_WAV  "/opt/test/test.wav"

#define MODEL_DIR "/opt/test/models/onnx_20211025_conformer_exp"
int main(int argc, char * argv[])
{


	auto Handle =RpASR_init(MODEL_DIR, RP_DEFAULT_THREAD_NUM);
	if (!Handle)
	{
		printf("Can't load models from %s\n", MODEL_DIR);
		return -1;
	}

	auto retHandle =RpASRRecogFile(Handle, TEST_WAV, RPASRM_ATTENSION_RESCORING); // RPASRM_CTC_GREEDY_SEARCH); // RPASRM_ATTENSION_RESCORING);
	int nNumber =RpASRGetRetNumber(retHandle);
	printf(" %d results.  String:", nNumber);
	const char * szString =RpASRGetResult(retHandle, 0);
	printf(szString);
	printf("\n");
	if (retHandle)
		RpASRFreeResult(retHandle);
	RpASR_Uninit(Handle);
	return 0;


}