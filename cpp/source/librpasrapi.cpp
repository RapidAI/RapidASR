#include "precomp.h"

#ifdef __cplusplus 

extern "C" {
#endif


	// APIs for rapidasr


	_RAPIDASRAPI RAPIDASR_HANDLE RpASR_init(const char* szModelDir,int nThread)
	{
		
		CQmASRRecog* pObj = new CQmASRRecog(szModelDir, nThread);
		if (pObj)
		{
			if (pObj->IsLoaded())
				return pObj;
			else
				delete pObj;

		}
	
		return nullptr;
	}


	_RAPIDASRAPI RAPIDASR_RESULT RpASRRecogBuffer(RAPIDASR_HANDLE handle, const char* szBuf, int nLen, RAPIDASR_MODE Mode)
	{

		
		CQmASRRecog* pRecogObj = (CQmASRRecog*)handle;

		if (!pRecogObj)
			return nullptr;
		vector<float>  wav;
	
		wenet::WavReaderMem Reader(szBuf,nLen, wav);

		assert(Reader.sample_rate() == Reader.sample_rate());
		wenet::FeaturePipelineConfig config(QM_FEATURE_DIMENSION, QM_DEFAULT_SAMPLE_RATE); 
		vector<vector<float>> feats;
		if (pRecogObj->ExtractFeature(wav, feats, config) > 0)
			return pRecogObj->DoRecognize(feats, Mode);
		else
			return nullptr;
	}

	_RAPIDASRAPI RAPIDASR_RESULT RpASRRecogFile(RAPIDASR_HANDLE handle, const char* szWavfile, RAPIDASR_MODE Mode)
	{
		CQmASRRecog* pRecogObj = (CQmASRRecog*)handle;

		if (!pRecogObj)
			return nullptr;
				
		vector<float>  wav;
		wenet::WavReader Reader(szWavfile, wav);
		assert(Reader.sample_rate() == Reader.sample_rate());
		wenet::FeaturePipelineConfig config(QM_FEATURE_DIMENSION, Reader.sample_rate());
		vector<vector<float>> feats;
		if (pRecogObj->ExtractFeature(wav, feats,config) > 0)
			return pRecogObj->DoRecognize(feats,Mode);
		else
			return nullptr;
	}

	_RAPIDASRAPI const int RpASRGetRetNumber(RAPIDASR_RESULT Result)
	{
		if (!Result)
			return 0;
		PRAPIDASR_RECOG_RESULT pResult = (PRAPIDASR_RECOG_RESULT)Result;
		return pResult->Strings.size();
		
	}
	_RAPIDASRAPI const char* RpASRGetResult(RAPIDASR_RESULT Result,int nIndex)
	{
		PRAPIDASR_RECOG_RESULT pResult = (PRAPIDASR_RECOG_RESULT)Result;
		if(!pResult)
			return nullptr;
		if (nIndex >= pResult->Strings.size())
			return nullptr;
		return pResult->Strings[nIndex].c_str();
	}

	_RAPIDASRAPI void RpASRFreeResult(RAPIDASR_RESULT Result)
	{

		if (Result)
		{
			delete PRAPIDASR_RECOG_RESULT(Result);

		}
	}

	_RAPIDASRAPI void RpASR_Uninit(RAPIDASR_HANDLE handle)
	{

		CQmASRRecog* pRecogObj = (CQmASRRecog*)handle;

		if (!pRecogObj)
			return;

		delete pRecogObj;

	}



#ifdef __cplusplus 

}
#endif

