

```
初始化程序库
_RAPIDASRAPI RPASR_HANDLE  RapidAsrInit(const char* szModelDir, int nThread);



// if not give a fnCallback ,it should be NULL 
识别内存缓冲区，完整的wav文件数据，包括文件头
_RAPIDASRAPI RPASR_RESULT	RapidAsrRecogBuffer(RPASR_HANDLE handle, const char* szBuf, int nLen, RPASR_MODE Mode, QM_CALLBACK fnCallback);
识别内存缓冲区，只包括采样点数据，不包括wav文件头
_RAPIDASRAPI RPASR_RESULT	RapidAsrRecogPCMBuffer(RPASR_HANDLE handle, const char* szBuf, int nLen, RPASR_MODE Mode, QM_CALLBACK fnCallback);
识别文件，只包括采样点数据，不包括wav文件头
_RAPIDASRAPI RPASR_RESULT	RapidAsrRecogPCMFile(RPASR_HANDLE handle, const char* szFileName, RPASR_MODE Mode, QM_CALLBACK fnCallback);

识别音频文件，完整的wav文件数据，包括文件头
_RAPIDASRAPI RPASR_RESULT	RapidAsrRecogFile(RPASR_HANDLE handle, const char* szWavfile, RPASR_MODE Mode, QM_CALLBACK fnCallback);

获取识别后的文本和相关数据
_RAPIDASRAPI const char*	RapidAsrGetResult(RPASR_RESULT Result,int nIndex);

获取结果块个数
_RAPIDASRAPI const int		RapidAsrGetRetNumber(RPASR_RESULT Result);

释放返回的结果块内存
_RAPIDASRAPI void			RapidAsrFreeResult(RPASR_RESULT Result);


使用完成后清理程序库
_RAPIDASRAPI void			RapidAsrUninit(RPASR_HANDLE Handle);

获取结果块中的数据所表示的音频长度，单位秒
_RAPIDASRAPI const float	RapidAsrGetRetSnippetTime(RPASR_RESULT Result);


```
