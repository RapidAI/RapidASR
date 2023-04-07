#pragma once

class CQmASRRecog
{
private:

	Ort::Session* m_session_encoder=nullptr;
	Ort::Session* m_session_decoder = nullptr;
	Ort::Env envDecoder = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "QmASR_decoder");
	Ort::Env envEncoder = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "QmASR_encoder");
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();

	string m_strConfig, m_strDict;

	vector<string> m_vecEncInputName, m_vecEncOutputName, m_vecDecInputName, m_vecDecOutputName;
	vector<const char *> m_strEncInputName, m_strEncOutputName, m_strDecInputName, m_strDecOutputName;
	bool m_bIsLoaded = false;
	vector<std::string> m_Vocabulary;

	float m_reverse_weight=0.f;  // 从train.yaml中读取

public :
	CQmASRRecog(const char * szModelDir, int nThread);
	CQmASRRecog(const char* szEncoder, const char* szDecoder, const char* szDict, const char* szConfig, int nThread);
	~CQmASRRecog();

	bool LoadModel(const char* szEncoder, const char* szDecoder, const char* szDict, const char* szConfig, int nNumThread);
	bool LoadModel(const char* szModelDir, int nNumThread);
	bool IsLoaded();

	int  ExtractFeature(vector<float>& wav, std::vector<std::vector<float>>& feats ,wenet::FeaturePipelineConfig& config); // 特征提取
	PRAPIDASR_RECOG_RESULT DoRecognize(vector<vector<float>>& feats, RAPIDASR_MODE Mode = RPASRM_CTC_GREEDY_SEARCH);


};