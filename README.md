# Label Attention을 이용한 영문 순차 태깅
 전북대학교 문헌정보학과 201518611 정민교

   실험결과
   
   1. Dev Set
   
  | Model                    | Precision | Recall | F1    |
|--------------------------|-----------|--------|-------|
| BI-LSTM-CRF              | 89.14     | 89.36  | 89.25 |
| BI-LSTM(Label Attention) | 90.34     | 90.02  | 90.18 |


   2. Test Set
| Model                    | Precision | Recall | F1   |
|--------------------------|-----------|--------|------|
| BI-LSTM-CRF              | 89.94     | 84.26  | 84.6 |
| BI-LSTM(Label Attention) | 85.13     | 83.69  | 84.4 |
