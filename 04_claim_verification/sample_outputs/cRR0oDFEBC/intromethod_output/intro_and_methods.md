# Title

SELF-PLAY WITH EXECUTION FEEDBACK: IMPROVINGINSTRUCTION-FOLLOWING CAPABILITIES OF LARGELANGUAGE MODELS


---


# Introduction

The instruction-following ability of large language models (LLMs) refers to their capacity to understand, interpret, and execute commands given to them in natural language (Lou et al., 2023; OpenAI et al., 2024; Yang et al., 2024a;b). This ability is fundamental to contemporary LLMs as it enables them to leverage their underlying knowledge, interact intuitively with users (Ouyang et al., 2022), adapt to various requirements (Zhang et al., 2023; Lei et al., 2023), and perform complex tasks (Sun et al., 2024; Dong et al., 2024b) and scenarios (Lu et al., 2024; Qiao et al., 2024a; Zhang et al., 2024). Misunderstandings in following instructions can lead to unintended outcomes, potentially resulting in severe consequences, particularly in critical scenarios (Zhou et al., 2023; Chang et al., 2024).

Although instruction following is crucial, scalable and reliable methods to enhance this capability of LLMs remain elusive. Current efforts in this field are divided into manual annotation (Wei et al., 2021; Zhou et al., 2023; Jiang et al., 2024b) and behavior imitation (Xu et al., 2023; Zhao et al., 2024). Manual annotation involves annotators designing instructions and writing corresponding responses. However, due to human cognition’s limitations, creating highly complex and diverse instructions is challenging, making the process difficult to scale. Furthermore, accurately executing complex instructions can sometimes be difficult for humans (Sun et al., 2024; Cao et al., 2024b; Hui et al., 2024), requiring multiple rounds of rigorous and costly validation (Wang et al., 2024a; Wei et al., 2024). On the other hand, behavior imitation aims to distill responses from more advanced LLMs (Taori et al., 2023; Wang et al., 2024b; Peng et al., 2023) like GPT-4. This approach limits models to the capabilities of the advanced LLMs from which they are distilled. Moreover, even advanced LLMs can make mistakes, and the reliability of the distilled data cannot be guaranteed (Cui et al., 2023). Consequently, models trained with this data may have a propensity to not follow instructions accurately (Zhou et al., 2024).

In this paper, we introduce AUTOIF, the first scalable and reliable method for automatically generating instruction following training Data for Supervised Finetuning (SFT) or Reinforcement Learning from Human Feedback (RLHF) (Ouyang et al., 2022). The core idea of AUTOIF is to use code to verify the correctness of following instructions. Intuitively, if designed properly, a significant portion of instructions, such as “Keep your response under 20 characters in length” can be verified for correctness using code, as illustrated in Fig. 1. Therefore, the key components of AUTOIF include (1) automatically generating instructions that can be verified by code, (2) automatically generating corresponding verification codes for these instructions, and (3) ensuring the reliability of the first two steps. Specifically, we start by providing AUTOIF with a small set of hand-written seed instructions. Then, LLMs, not necessarily advanced ones, generate an augmented instruction set through self-instruct (Wang et al., 2023a; Li et al., 2024b). Next, LLMs write verification codes and unit test cases for each instruction. Only the code that compiles correctly, passes the test cases, and back-translates to the original instruction is retained. If an instruction does not have a corresponding code that can verify its correctness, it is discarded. Finally, we employ LLMs to generate responses that either pass or fail the verification code using execution feedback-based rejection sampling (Yuan et al., 2023). Responses that pass can be directly used for SFT, while pairs of passing and failing responses can be naturally used to create chosen-rejected pairs for Direct Preference Optimization (DPO) (Rafailov et al., 2023) and other RLHF algorithms. Moreover, once the instructions and verification code are determined, this process can be conducted on-policy, iteratively enhancing the instruction-following capabilities.

Through extensive experiments, we demonstate that AUTOIF significantly improves performance across three training algorithms—SFT, Offline DPO, and Online DPO—when applied to leading open-source LLMs, Qwen2-72B and LLaMA3-70B, in both self-alignment and strong-to-weak distillation settings. We conduct a comprehensive evaluation of five general instruction-following datasets, verfying AUTOIF’s strong general instruction alignment capabilities. Notably, we first achieve Loose Instruction accuracy rates of $8 8 . 0 \%$ with Qwen2-72B and $9 0 . 4 \%$ with LLaMA3-70B on IFEval, the most widely used instruction-following benchmark, while significantly preserving the LLM’s coding, mathematical, and general interaction capabilities. We will open-source the SFT and DPO datasets and construction codes built with AUTOIF on Qwen2-72B, marking the first large-scale, complex instruction-following dataset of its kind.

To summarize, our contributions are as follows:

• To achieve automated, reliable improvement of LLMs’ instruction-following with minimal human efforts, we propose AUTOIF, which first transforms instruction-following alignment into automatically code verification, requiring LLMs to generate instructions, corresponding verification code, and unit test samples for cross-validation.   
• Based on DPO algorithms, we first regard executor feedback as a natural reward model, constructing pairwise preference samples from both instruction and query aspects. We further design offline and on-policy strategies for iterative optimization of the model’s weakness on instruction following.   
• With AUTOIF, we validate AUTOIF’s effectiveness in both "Self-Alignment" and "Strong-to-Weak" settings on two widely used IF benchmarks and three general IF benchmarks, especially first achieving over $90 \%$ accuracy in IFEval’s Loose instruction Acc without compromising general abilities, math, and code reasoning. Further analysis on quality, scaling, combination, and data efficiency showcases AutoIF’s robust generalization and alignment potential.


---


# Methods

# 3 AUTOIF

We introduce AUTOIF, an automated, scalable, and reliable method designed to enhance the instruction-following capabilities of LLMs. In this section, we outline the preliminaries (§3.1), detail the two core components of AUTOIF (§3.2, §3.3), and discuss various training strategies that can be seamlessly integrated with AUTOIF (§3.4).

# 3.1 PRELIMINARIES

Instruction-following Capabilities. Following instructions is one of the most crucial skills in modern LLMs. These models are expected to provide precise responses to queries containing

![](images/962e7d0d87bd8ddb6430b3227157811464e73f187b1605a4cdb61b850061f676.jpg)

![](images/320170cc7cb0103c2cf9bfff231c1a452b65cd20094f157a15ae904a98054e84.jpg)  
Figure 2: An Overview of AUTOIF: An Automated Instruction-Following Data Synthesis Method.

complex instructions, which can be either atomic or compositional. To evaluate the instructionfollowing capability of LLMs, we define a general instruction-following requirement as a specific task. In this task, given an instruction $I = \{ i _ { j } \} _ { j = 1 } ^ { N }$ with $N$ specific constraints (e.g. “Please generate text in Shakespearean style, no more than 50 tokens” contains 2 constraints) and a specific query $x$ , an LLM $\pi _ { \theta }$ should generate precise response $y \sim \pi _ { \boldsymbol { \theta } } ( y \mid x , I )$ adhering to the constraints.

Verifiable Instructions. The complexity and diversity of instructions necessitate manual construction and verification for reliable supervision. This practical challenge motivates us to focus initially on instructions that can be automatically verified through programs and code executors, also known as verifiable instructions (Zhou et al., 2023). Specifically, for a given instruction $I$ and task-specific query $q$ , there exists a verification function $f _ { I }$ such that $f _ { I } ( y )$ returns true when the model’s response $y$ correctly follows the instruction. We demonstrate that supervision of such instructions can be self-generated through scalable oversight with LLMs and execution feedback. Extensive experiments in our work show that training on verifiable instructions significantly benefits the handling of other general instructions that are more complex but unverifiable with simple code snippets.

Method Overview. AUTOIF synthesizes high-quality instruction-following data through selfevolution, rejection sampling, and execution feedback. As illustrated in Fig. 2, AUTOIF integrates automated data augmentation with quality verification processes, including automatically generated verification functions and back-translation instructions. This approach enables a two-stage automated data synthesis at both the instruction (§3.2) and query levels (§3.3). Additionally, we introduce three training strategies (§3.4) and explore two experimental settings (§4) to thoroughly evaluate the effectiveness and generalization of AUTOIF.

# 3.2 INSTRUCTION AUGMENTATION AND VERIFICATION

We first develop verifiable instructions along with corresponding evaluation functions, using rejection sampling informed by execution feedback.

Seed Instruction Construction. We start by handwriting a set of seed instructions, denoted as $D _ { s e e d }$ , ensuring that each instruction contains only a single atomic constraint (e.g., “Answer the words that begin with B”). Detailed information on seed instructions is listed in Appx. $\ S C$ .

Self-Instruct. Self-Instruct (Wang et al., 2023a) is a straightforward and intuitive strategy for automated data augmentation that has garnered significant attention in the field of LLM reasoning (Xu et al., 2023; Zhao et al., 2023). For each instruction in $D _ { s e e d }$ , we use an LLM to perform $K$ instruction rewrites, generating $D _ { a u g }$ . We then combine the seed and augmented data sets to obtain an enhanced set of instructions, $D _ { i n s } \mathsf { \bar { = } } D _ { s e e d } \cup D _ { a u g }$ , and remove any duplicates.

Automated Quality Cross Verification. Previous research has shown that relying solely on modelgenerated augmented instructions often leads to the inclusion of low-quality samples (Bai et al., 2022; Mumuni & Mumuni, 2022; Dong et al., 2024c; Xie et al., 2020; Zheng et al., 2024). Inspired by a

![](images/a169443b9b9b0e0e70e3d1ee518f48f7d6cb47325aff0f972267a12bdad0c74f.jpg)

![](images/6a391479480e9456272cfe7f6ad43db208c1e274decbbc4a367de964918aa4da.jpg)  
Figure 3: Different training strategies that can be adapted with synthetic dataset generated by AUTOIF.

series of tool execution studies, we employ an LLM to generate verification functions and test cases for each instruction. We use feedback from executing Python programs to ensure quality control. Given the instruction set $D _ { i n s }$ , the LLM $M$ employs a rejection sampling (Touvron et al., 2023; Yuan et al., 2023) to generate $K$ verification functions $f _ { I } = \stackrel { \bf \bar { \theta } } { \left\{ f _ { i } \right\} } _ { i = 1 } ^ { K }$ and test cases $c _ { I } = \{ c _ { i } \} _ { i = 1 } ^ { K }$ for each instruction $I$ , resulting in the set $\{ I , f _ { I } , c _ { I } \} \in D _ { i n s }$ . We then cross-validate the quality of the instructions using the verification functions and test cases, ensuring they meet the following criteria:

• The verification function $f \in f _ { I }$ can be successfully compiled by the Python executor.   
• Each test case $c \in c _ { I }$ achieves an accuracy rate greater than 0.5 across all verification functions.   
• Each verification function $f \in f _ { I }$ achieves an accuracy rate greater than 0.5 across all test cases.   
• Each instruction includes at least one evaluation function and test case.

By adhering to these four conditions, we obtain the quality-filtered instruction set $\{ I ^ { ( 2 ) } , f _ { I } ^ { ( 2 ) } \} \in D _ { i n s } ^ { ( 2 ) }$ .

Back-translation Verification. After the cross-validation stage, we obtained initially quality-verified verification functions and instructions. To further ensure the consistency between instructions and verification functions, we introduce back-translation. For a given pair $\{ I ^ { ( 2 ) } , f _ { I } ^ { ( 2 ) } \} \in D _ { i n s } ^ { ( 2 ) }$ , we use the LLM $M$ to back-translate the verification function $f \in f _ { I } ^ { ( 2 ) }$ into instruction $I _ { f }$ . We then treat $I$ as the premise and the back-translated instruction $I _ { f }$ as the hypothesis. Using the NLI model, we identify the semantic relationship between the two instructions. The prediction can fall into one of three categories: entailment, contradiction, or neutral:

$$
p _ {\theta} (\cdot \mid q, q _ {\mathrm {a u g}}) = \operatorname {s o f t m a x} \left(\operatorname {s c o r e} _ {\theta} (I, I _ {f})\right), \tag {1}
$$

where $\mathrm { s c o r e } _ { \theta } : \mathbb { R } ^ { k \times \ell _ { I } } \times \mathbb { R } ^ { k \times \ell _ { I _ { f } } } \to \mathbb { R } ^ { 3 }$ is a model dependent scoring function with parameters $\theta$ . We filter out any instruction $I$ labeled as contradiction to ensure the intent consistency. Finally we obtain the set {I (3), f (3)I } ∈ D(3)ins $\{ I ^ { ( 3 ) } , f _ { I } ^ { ( 3 ) } \} \in D _ { i n s } ^ { ( 3 ) }$

# 3.3 QUERY AUGMENTATION AND VERIFICATION

Once we have obtained verified instructions and verification functions, we utilize them to create training data comprising queries and responses.

Query Reforming and Augmentation. In the real-world application of modern chatbots, instructions are typically employed to generate constrained responses to user queries. Therefore, creating high-quality instructions is merely the initial step toward achieving effective instruction-following capabilities. To acquire authentic queries, as shown in the bottom part of Fig. 2, we randomly selected $K$ user queries from ShareGPT (Chiang et al., 2023) for each instruction and concatenated them to construct the seed query dataset $x , f _ { I } ^ { ( 3 ) } \in D _ { q }$ . To further enhance the diversity and complexity of the input $x$ , we utilized the LLM to generate $K$ responses $y _ { x } = \{ y _ { i } \} _ { i = 1 } ^ { K }$ , resulting in $\{ x , f _ { I } ^ { 3 } , y _ { x } \} \in D _ { q }$ .

Instruction-following Verification. Following the previous quality cross-verification process, we further employ verification functions to assess whether the augmented responses adhere to the constraints in input $x$ . Similarly, we require each response in $D _ { q }$ to meet the following conditions:

• Each response must achieve an accuracy rate greater than 0.5 across all verification functions.   
• Each input must include at least one verification function and one response.

Based on these rules, we obtain the set $( x ^ { ( 2 ) } , f _ { I } ^ { ( 3 ) } , y ^ { ( 2 ) } ) \in D _ { q } ^ { ( 2 ) } .$ .

Query Quality Verification. Additionally, we observe that concatenated instructions and queries often conflict. For instance, a high-quality response to the query “help me write a news article” is unlikely to comply with the instruction “please limit your answer to two words”. Such high-level semantic inconsistencies are challenging for a simple NLI model to discern. Therefore, we employ the LLM $M$ to assign matching scores between the instruction and query in input $x ^ { ( 2 ) }$ and the corresponding responses $y ^ { ( 2 ) }$ , on a scale from 1 to 10. We then filter out samples with a score lower than 8, constructing the final training set $D _ { \mathrm { t r a i n } } = \{ x _ { i } , y _ { i } , f _ { I i } \} _ { i = 1 } ^ { N }$ .

# 3.4 TRAINING STRATEGIES

AUTOIF offers multifaceted supervision for the instruction-following task, making it adaptable to various training strategies. To thoroughly evaluate the effectiveness of AUTOIF, we propose the following training approaches:

Supervised Fine-tuning (SFT). Given $( x _ { i } , y _ { i } ) \in D _ { \mathrm { f i n a l } }$ , we apply the standard Supervised Finetuning (SFT) objective on the base model $P$ with parameters θ: $\begin{array} { r } { : \bar { \mathcal { L } ( \pmb { \theta } ) } = \sum _ { ( x _ { i } , y _ { i } ) \in \mathcal { D } _ { \operatorname { t r a i n } } } \bar { \log \mathbb { P } } _ { \pmb { \theta } } ( y _ { i } \mid x _ { i } ) } \end{array}$ , where $x _ { i }$ denotes the $i$ -th input, consisting of a concatenated instruction and user query.

SFT $^ +$ Offline DPO. In the process of AUTOIF, multiple scales of quality filtering are utilized, naturally generating a substantial number of positive and negative sample pairs. This motivates us to obtain pairwise preference data $( x , y _ { w } , y _ { l } )$ . Our preference data mining is divided into two parts:

• Instruction Level: During the automated quality cross-verification stage, we first extract positive samples $c _ { w }$ from cases with an accuracy rate higher than 0.5 on all verification functions and negative samples $c _ { l }$ from cases with an accuracy rate of 0. We then construct pairwise preference data for each instruction: $D _ { \mathrm { i n s } } ^ { \mathrm { p r e f } } \to \left( I , c _ { w } , c _ { l } \right)$ .   
• Query Level: In the query quality verification process, we similarly extract positive samples $y _ { w }$ from responses with an accuracy rate higher than 0.5 on all verification functions and negative samples $D _ { \mathsf { q u e r y } } ^ { \mathsf { p r e f } } \to ( x , y _ { w } , y _ { l } )$ $y _ { l }$ from responses with an accuracy rate of 0. We then construct query preference data: .

Finally, we merge the two parts of the data: $D _ { \mathrm { p r e f } } = D _ { \mathrm { i n s } } ^ { \mathrm { p r e f } } \cup D _ { \mathrm { q u e r y } } ^ { \mathrm { p r e f } }$ . To further explore the potential of pairwise preference data $( x , y _ { w } , y _ { l } ) \in \mathop { D _ { \mathrm { p r e f } } } ^ { }$ , we first perform vanilla SFT on the base model $\pi _ { \theta }$ to obtain an SFT model $\pi _ { \theta } ^ { \mathrm { S F T } }$ as equation 3.4. Then, we apply Direct Preference Optimization (DPO) (Rafailov et al., 2024) on our SFT model, which can be formulated as follows:

$$
\mathcal {L} _ {\mathrm {D P O}} \left(\pi_ {\theta} ^ {\mathrm {S F T}}; \pi_ {\mathrm {r e f}}\right) = - \mathbb {E} _ {\left(x, y _ {w}, y _ {l}\right) \sim \mathcal {D}} \left[ \log \sigma \left(\beta \log \frac {\pi_ {\theta} ^ {\mathrm {S F T}} \left(y _ {w} \mid x\right)}{\pi_ {\mathrm {r e f}} \left(y _ {w} \mid x\right)} - \beta \log \frac {\pi_ {\theta} ^ {\mathrm {S F T}} \left(y _ {l} \mid x\right)}{\pi_ {\mathrm {r e f}} \left(y _ {l} \mid x\right)}\right) \right], \tag {2}
$$

where the reference ma hyperparameter and el is $\pi _ { \mathrm { r e f } }$ is set to  sigmoid $\pi _ { \theta } ^ { \mathrm { S F T } }$ initiation. d remains fixed throughout training. aims to maximize the log probabilit $\beta$ is of $\sigma$ $\mathcal { L } _ { \mathrm { D P O } }$ preferred $y _ { w }$ relative to the dispreferred $y _ { l }$ .

SFT $^ +$ Iterative Online DPO. Online training enables real-time, iterative optimization of model weaknesses. It relies on high-quality, lightweight reward models to provide continuous supervision feedback. In the case of AUTOIF, verification functions serve as rigorous filtering standards, akin to reward models, delivering immediate feedback on model responses across training iterations. Following offline DPO, we conduct initial SFT on the base model with initial instruction-following capabilities. As depicted in Fig. 3, $\pi _ { \theta }$ to derive an SFT model  set the generation temper $\pi _ { \theta } ^ { \mathrm { S F T } }$

to 0.8 and allow the SFT model to generate $K$ responses through self-sampling for each training sample, forming a response set $\{ R _ { 1 } , \ldots , R _ { k } \}$ . Then, we employ corresponding verification functions to assess K responses, thereby constructing the online DPO dataset Dprefonline $K$ $D _ { \mathrm { o n l i n e } } ^ { \mathrm { p r e f } } = ( x , y _ { w } , y _ { l } )$ based on average pass rates across all functions. Finally, leveraging $D _ { \mathrm { o n l i n e } }$ , we sequentially perform DPO training on $\pi _ { \theta } ^ { \mathrm { S F T } }$ . Importantly, our iterative online optimization process progressively unlocks enhanced instruction-following capabilities.