# 다층 하이브리드 연합학습 기반 분산형 데이터 분석 및 예측 모델 개발
## 김은석(전남대학교 데이터사이언스대학원 데이터사이언스학과)

본 연구는 개인정보 보호와 모델 성능을 동시에 달성하기 위한 다층적 하이브리드 연합학습 시스템을 설계하고, 극단적 데이터 희소성 환경에서의 효과를 실증적으로 검증하였다. General Data Protection Regulation(GDPR)과 Health Insurance Portability and Accountability Act(HIPAA) 등 강화된 개인정보 보호 규제 환경에서 중앙집중식 학습의 한계를 극복하고자, 신뢰도 기반 가중치 메커니즘과 계층적 집계 구조를 결합한 차세대 연합학습 아키텍처를 제안하였다.

세 가지 의료 데이터셋(PIMA Indians Diabetes, Maternal Health and High-Risk Pregnancy, UCI Heart Disease)과 일곱 가지 머신러닝 모델(Logistic Regression, LSTM, Random Forest, SVC, VAE, SecureBoost, LightGBM)을 대상으로 클라이언트 수를 100명에서 1,300명까지 확장하며 총 273개의 실험 시나리오를 수행하였다. 클라이언트당 평균 데이터가 1개 미만인 극단적 희소성 환경을 의도적으로 조성하여 모델의 극한 상황 대응 능력을 평가하였다.

신경망 기반 모델에서 탁월한 성능 개선이 확인되었다. LSTM은 평균 정확도 1.38%, 정밀도 1.46%, 재현율 1.27%, F1 점수 1.32% 향상을 달성하였으며, 특히 400, 800, 1,100 클라이언트 구간에서 최대 5%p의 성능 차이를 보였다. VAE는 정확도 1.42%, 재현율 1.88% 개선을 기록하여 실제 환자 누락을 감소시키는 임상적 활용 가능성을 시사한다. 반면 Random Forest, LightGBM, SecureBoost 등 트리 기반 모델은 다층 구조의 이점을 활용하지 못하거나 오히려 성능 저하를 보여, 모델 특성에 따른 차별화된 최적화 전략의 필요성을 확인하였다.

극단적 데이터 희소성 환경에서 다층적 방식의 시스템 붕괴 방어 능력이 입증되었다. PIMA 데이터셋 600명 클라이언트 구간에서 기존 방식은 정확도 0.3931로 급락한 반면 다층적 방식은 0.5706을 유지하였으며, 1,100명 구간에서는 0.6040 대 0.7610으로 약 26.0%의 유의미한 성능 향상을 확인하였다. 이는 최근 10라운드 참여 성공률 기반 동적 신뢰도 계산과 0.3~1.2 범위의 비선형 가중치 함수가 저품질 클라이언트를 효과적으로 억제하고 고품질 클라이언트 중심으로 학습을 주도하게 한 결과이다.

의료 AI로서의 임상적 신뢰성도 확보하였다. 정밀도 3.91% 향상은 건강한 사람을 환자로 오진하는 위양성을 감소시켜 불필요한 의료 개입과 환자 심리적 부담을 경감하였고, 재현율 최대 1.88% 향상은 실제 환자 조기 발견 능력을 강화하여 만성질환 합병증 감소에 기여할 수 있음을 보였다. F1 점수의 균형적 개선은 클래스 불균형 환경에서도 안정적인 screening tool로 활용 가능함을 입증하였다.

본 연구는 신뢰도 기반 집계가 단순한 성능 향상 기법을 넘어 Byzantine-Resilient 특성을 자연스럽게 발현하며, Non-IID 환경에서 계층적 완충 효과를 통해 안정성과 확장성을 동시에 확보하는 근본적 설계 원칙임을 규명하였다. 프라이버시 보존과 모델 성능의 양립 가능성을 실증하여 GDPR과 HIPAA 준수 환경에서도 최대 91.10%의 정확도를 달성함으로써, 의료 자원이 부족한 지역에서 AI 기반 조기 진단 도구로 활용될 수 있는 기술적 토대를 마련하였다.

# Multi-Layered Hybrid Federated Learning-Based Distributed Data Analysis and Predictive Model Development
## Eunseok Kim(Master of Data Science, Graduate School of Data Science, Chonnam National University)

This study designed a multi-layer hybrid federated learning system to simultaneously achieve privacy protection and model performance, and empirically verified its effectiveness in extreme data scarcity environments. To overcome the limitations of centralized learning under strengthened privacy regulations such as General Data Protection Regulation(GDPR) and Health Insurance Portability and Accountability Act(HIPAA), we proposed a next-generation federated learning architecture combining trust-based weighting mechanisms and hierarchical aggregation structures.

We conducted 273 experimental scenarios with three medical datasets (PIMA Indians Diabetes, Maternal Health and High-Risk Pregnancy, UCI Heart Disease) and seven machine learning models (Logistic Regression, LSTM, Random Forest, SVC, VAE, SecureBoost, LightGBM), scaling from 100 to 1,300 clients. We intentionally created extreme scarcity environments with less than one data sample per client on average to evaluate the models' resilience under extreme conditions.

Exceptional performance improvements were confirmed in neural network-based models. LSTM achieved improvements of 1.38% in average accuracy, 1.46% in precision, 1.27% in recall, and 1.32% in F1 score, with performance differences reaching up to 5 percentage points in the 400, 800, and 1,100 client ranges. VAE recorded improvements of 1.42% in accuracy and 1.88% in recall, demonstrating clinical value in reducing missed patient detections. Conversely, tree-based models such as Random Forest, LightGBM, and SecureBoost failed to leverage the advantages of multi-layer structures or even showed performance degradation, confirming the necessity of differentiated optimization strategies according to model characteristics.

The multi-layer approach's capability to prevent system collapse in extreme data scarcity was demonstrated. In the PIMA dataset with 600 clients, while the conventional approach plummeted to 0.3931 accuracy, the multi-layer approach maintained 0.5706. At 1,100 clients, it achieved an overwhelming performance superiority of approximately 26.0%, with 0.7610 versus 0.6040. This resulted from dynamic trust calculation based on the last 10 rounds' participation success rates and a nonlinear weight function ranging from 0.3 to 1.2, which effectively suppressed low-quality clients and led learning centered on high-quality clients.

Clinical reliability as medical AI was also secured. The 3.91% improvement in precision reduced false positives of misdiagnosing healthy individuals as patients, alleviating unnecessary medical interventions and patient psychological burden. The maximum 1.88% improvement in recall strengthened the ability to detect actual patients early, potentially contributing to reducing chronic disease complications. The balanced improvement in F1 scores demonstrated usability as a stable screening tool even in class-imbalanced environments.

This study identified that trust-based aggregation goes beyond simple performance enhancement techniques, naturally manifesting Byzantine-Resilient characteristics and simultaneously securing stability and scalability through hierarchical buffering effects in non-IID environments as a fundamental design principle. By demonstrating the compatibility of privacy preservation and model performance, achieving up to 91.10% accuracy in GDPR and HIPAA-compliant environments, we established a technological foundation for AI-based early diagnostic tools in resource-limited medical regions.
