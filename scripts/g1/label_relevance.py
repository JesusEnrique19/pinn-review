"""LLM labels for the relevance-classifier samples (pending human audit).

    python scripts/g1/label_relevance.py test
    python scripts/g1/label_relevance.py train

The test labels were written before any classifier existed and are never used for training.
"""
import sys

import pandas as pd

from pinnreview.validation import EXCEL_ENCODING, G1

Y, N = "y", "n"

TEST = {
0:(Y,"l","PINN-based polynomial chaos for an integral-equation EM method"),1:(Y,"h",""),2:(N,"l","GNN performance benchmarking, no DE solving"),
3:(Y,"h",""),4:(N,"l","generic hyperparameter sensitivity"),5:(N,"h",""),6:(N,"l","physics-informed RNN for bathymetry retrieval"),
7:(N,"l",""),8:(Y,"h",""),9:(Y,"h","circuit ODEs"),10:(N,"l","physics-informed data generation for medical small data"),11:(Y,"h",""),
12:(N,"l","fixed-point theory in fuzzy metric spaces"),13:(N,"l","mode decomposition with known eigenmodes"),14:(N,"l","Greek-language thesis"),
15:(Y,"h",""),16:(Y,"l","neural control for HJI-type problems"),17:(Y,"h",""),18:(Y,"h","Black-Scholes"),19:(N,"h",""),
20:(N,"l","learning geodesic deformation flows for images"),21:(Y,"l","overview of AI for PDEs"),22:(N,"h",""),23:(N,"h",""),24:(Y,"h",""),
25:(N,"h",""),26:(Y,"h",""),27:(Y,"h",""),28:(Y,"h",""),29:(N,"h",""),30:(Y,"h",""),31:(N,"l","FNO as data-driven damage classifier"),
32:(Y,"l","autoencoder identifying differential equations"),33:(Y,"l","ML acceleration of gravity-wave simulations"),34:(N,"h","ML interatomic potentials"),
35:(Y,"l","deep learning with population balance model"),36:(N,"l","PINN for CCHP system optimisation; DE unclear"),37:(Y,"h",""),38:(Y,"h",""),
39:(N,"h","theology; author surname Pinn"),40:(N,"l","NN-inspired Jacobi iteration"),41:(Y,"h",""),42:(Y,"l","physics-constrained NN with FE stiffness matrix"),
43:(Y,"h",""),44:(Y,"h",""),45:(Y,"h",""),46:(Y,"h",""),47:(Y,"h",""),48:(Y,"h",""),49:(Y,"h",""),50:(Y,"h",""),
51:(Y,"l","FNO for guided-wave imaging"),52:(N,"h",""),53:(N,"h","'PINN' for crop yield"),54:(Y,"l","PINN for reactor temperature trajectory"),
55:(Y,"h",""),56:(N,"l","device compact model"),57:(Y,"h",""),58:(Y,"h",""),59:(Y,"h",""),60:(Y,"h",""),
61:(Y,"l","neural computing for nanofluid transport ODEs"),62:(N,"h",""),63:(Y,"l","supervised NN for nanofluid ODE system"),64:(Y,"h",""),
65:(N,"l","physics-informed deep image prior for PET"),66:(Y,"h",""),67:(Y,"l","LSTM for parameters of a PDE epidemic model"),68:(N,"h",""),
69:(N,"l","flash calculation, algebraic"),70:(Y,"h",""),71:(Y,"h",""),72:(N,"h",""),73:(Y,"l","physics-guided network for structural dynamics"),
74:(Y,"h",""),75:(Y,"h",""),76:(Y,"h",""),77:(N,"l","physics-informed diffusion model for metasurface design"),78:(Y,"h",""),79:(Y,"h",""),
80:(Y,"h",""),81:(N,"l",""),82:(N,"l","resilience assessment; DE unclear"),83:(Y,"l","supervised NN for nanofluid flow equations"),84:(Y,"h",""),
85:(Y,"h",""),86:(Y,"h",""),87:(N,"l","fault diagnosis"),88:(Y,"h",""),89:(N,"l",""),90:(N,"h",""),91:(Y,"h",""),92:(Y,"h",""),
93:(Y,"h",""),94:(Y,"h",""),95:(Y,"h",""),96:(N,"l","entropy metrics for data-driven modelling"),97:(Y,"h",""),98:(N,"h",""),
99:(Y,"l","deep learning for nanofluid ODE system"),100:(N,"h","ML for stencil-code performance"),101:(Y,"h",""),102:(N,"h",""),103:(Y,"h",""),
104:(Y,"h",""),105:(Y,"h",""),106:(Y,"h",""),107:(Y,"l","data-driven PDE discovery"),108:(N,"l","battery resistance estimation"),109:(Y,"h",""),
110:(N,"l","path-loss prediction"),111:(Y,"h",""),112:(N,"l","physics-informed generative weather prediction"),113:(N,"h","finite differences, no NN"),
114:(Y,"h",""),115:(N,"h",""),116:(N,"h",""),117:(Y,"h",""),118:(Y,"h",""),119:(Y,"l","FNO for high-frequency ground motion"),120:(Y,"h",""),
121:(Y,"h",""),122:(N,"h",""),123:(Y,"l","physics-informed LSTM of electrochemical battery model"),124:(Y,"l","physics-informed RNN for nonlinear Schrodinger propagation"),
125:(Y,"h",""),126:(Y,"l","RBF NN for nanofluid boundary layer"),127:(Y,"l","PINN with neurodynamic ODE"),128:(Y,"h",""),129:(Y,"l","learning-based control of hyperbolic PDEs"),
130:(Y,"h",""),131:(Y,"h",""),132:(Y,"l","PINN-based control, ODE dynamics"),133:(Y,"h",""),134:(Y,"h",""),135:(N,"h",""),136:(Y,"h",""),
137:(N,"l","constrained DL option pricing; PDE unclear"),138:(N,"l","dissertation"),139:(N,"l",""),140:(Y,"h",""),141:(Y,"l","physics + data vehicle dynamics"),
142:(Y,"l","AI-driven hydrodynamic simulations"),143:(Y,"l","deep Kalman filter for hidden dynamics"),144:(Y,"h",""),145:(Y,"h",""),
146:(N,"l","physics-informed ML for membrane fouling"),147:(N,"l","FNO mapping scattering amplitudes, no DE"),148:(N,"l","data-driven flow super-resolution"),
149:(Y,"h",""),150:(Y,"h",""),151:(Y,"h",""),152:(Y,"h",""),153:(N,"l",""),154:(N,"h",""),155:(Y,"h",""),156:(N,"h",""),157:(Y,"h",""),
158:(N,"l",""),159:(Y,"l","neural simulation of nanofluid flow equations"),160:(Y,"h",""),161:(Y,"h",""),162:(Y,"h",""),
163:(Y,"l","physics-informed inversion with integral equations"),164:(Y,"l","ANN-POD reduced-order model"),165:(Y,"h",""),166:(Y,"h",""),167:(Y,"h",""),
168:(Y,"h",""),169:(N,"h",""),170:(Y,"h",""),171:(Y,"l","PINN critical-flow model"),172:(N,"l","numerical PDE for signatures"),173:(Y,"h",""),
174:(N,"l","empirical physics constraints"),175:(N,"l","generic model-based NN structures"),176:(Y,"h",""),177:(N,"l","power-flow (algebraic) PINN"),
178:(Y,"l","PINN parameter identification, ODE"),179:(N,"h",""),180:(Y,"h",""),181:(N,"h",""),182:(Y,"h",""),183:(Y,"l","generative neural physics with wave equation"),
184:(N,"l",""),185:(N,"h",""),186:(Y,"h",""),187:(N,"h","image synthesis"),188:(Y,"h",""),189:(Y,"h",""),190:(Y,"h",""),191:(N,"l",""),192:(Y,"h",""),
193:(N,"l","model-informed photoacoustic reconstruction"),194:(Y,"l","DL for stiff chemical kinetics"),195:(N,"l",""),196:(N,"l",""),197:(Y,"h",""),
198:(Y,"h",""),199:(Y,"h",""),
}

# Rule fixed while labelling: supervised NNs (e.g. Levenberg-Marquardt) that approximate the solution
# profile/field of a DE are in scope; NNs regressing scalar engineering outputs are not. Both are "low".
TRAIN = {
0:(Y,"h",""),1:(Y,"h",""),2:(Y,"h",""),3:(Y,"h",""),4:(Y,"h",""),5:(N,"l","autodiff meshfree interpolation without NN"),6:(Y,"h",""),
7:(Y,"h",""),8:(Y,"h",""),9:(N,"l",""),10:(Y,"h",""),11:(Y,"l","physics-constrained surrogate"),12:(N,"l",""),13:(Y,"l","DL for bioconvection flow equations"),
14:(Y,"l","neural interface reconstruction for EIT inverse problem"),15:(Y,"h",""),16:(Y,"l","review of GNN surrogates for reservoir PDEs"),
17:(Y,"l","cross-diffusion digital twin"),18:(N,"l",""),19:(Y,"l","PIML flow prediction for airfoil design"),20:(Y,"l","FWI with deep generative prior"),
21:(N,"l","aeroengine performance model, likely algebraic"),22:(Y,"h",""),23:(N,"h",""),24:(Y,"h",""),25:(Y,"h",""),26:(N,"l","power-grid attack localisation"),
27:(Y,"h","DL surrogate of RANS fields"),28:(Y,"h",""),29:(N,"l",""),30:(Y,"h",""),31:(Y,"h",""),32:(N,"l","generic KAN architecture"),33:(N,"l",""),
34:(Y,"h","orbit ODE"),35:(Y,"l","NN approximation for SQG equations"),36:(Y,"l","sub-equation NN for exact PDE solutions"),37:(Y,"l","PIML melt-pool dynamics"),
38:(N,"l",""),39:(Y,"l","deep fictitious play for stochastic differential games"),40:(Y,"h",""),41:(Y,"h",""),42:(Y,"l","PINN with electrochemical PDE model"),
43:(Y,"h",""),44:(Y,"h",""),45:(Y,"h",""),46:(N,"l",""),47:(Y,"h",""),48:(Y,"h",""),49:(Y,"h",""),50:(Y,"h",""),51:(N,"l","DCE-MRI parameter regression"),
52:(N,"l",""),53:(Y,"h",""),54:(N,"h","quantum computing for DAEs, no NN"),55:(Y,"h",""),56:(N,"h",""),57:(Y,"h",""),58:(Y,"l","PINN for protein folding dynamics"),
59:(Y,"l","PIML review for biomolecular dynamics"),60:(Y,"h",""),61:(Y,"l","symmetry discovery for governing equations"),62:(N,"h",""),63:(Y,"h",""),
64:(Y,"l","physics-informed data assimilation"),65:(Y,"h",""),66:(Y,"h",""),67:(Y,"h",""),68:(N,"h",""),69:(N,"l","adaptive dynamic programming control"),
70:(N,"l",""),71:(N,"l","pedotransfer functions"),72:(Y,"l","DNN for dusty-fluid flow equations"),73:(Y,"h",""),74:(Y,"h",""),75:(N,"l",""),
76:(Y,"l","physics-informed DL radar inversion"),77:(Y,"h",""),78:(N,"h","FNO for image reconstruction"),79:(Y,"h",""),80:(N,"h",""),
81:(N,"l","LS-SVM, not a neural network"),82:(N,"l","MRI reconstruction"),83:(N,"l","drilling fault diagnosis"),84:(N,"l","neural operator as damage classifier"),
85:(Y,"h",""),86:(N,"l",""),87:(Y,"h",""),88:(N,"l","image registration with mechanical regulariser"),89:(Y,"l","adaptive sampling for temperature-field PINN"),
90:(N,"l",""),91:(Y,"h","eikonal"),92:(Y,"h",""),93:(Y,"h",""),94:(N,"l",""),95:(Y,"h",""),96:(Y,"h",""),97:(N,"l","operator inference is regression, not NN"),
98:(Y,"h",""),99:(Y,"h",""),100:(N,"h",""),101:(N,"h",""),102:(Y,"h",""),103:(Y,"h",""),104:(Y,"h",""),105:(N,"l",""),
106:(N,"l","ANN for geometric constants"),107:(Y,"h",""),108:(Y,"h",""),109:(N,"h",""),110:(Y,"h",""),111:(Y,"l","LM-ANN for nanofluid flow equations"),
112:(N,"l","battery degradation"),113:(Y,"h",""),114:(Y,"h",""),115:(N,"l",""),116:(Y,"l","ML-based optimisation with PDE reduced models"),
117:(N,"l",""),118:(Y,"h",""),119:(N,"h",""),120:(N,"l",""),121:(Y,"l","PINN heat-loss model"),122:(N,"l",""),123:(N,"h",""),124:(Y,"h",""),
125:(N,"h",""),126:(Y,"h",""),127:(N,"l",""),128:(N,"h",""),129:(Y,"h",""),130:(Y,"h",""),131:(N,"l","empirical correlations"),
132:(Y,"l","DL flow and heat-transfer simulation"),133:(Y,"h",""),134:(Y,"h",""),135:(Y,"h",""),136:(Y,"h",""),137:(N,"h",""),138:(Y,"h",""),
139:(Y,"h",""),140:(N,"l",""),141:(N,"l",""),142:(N,"h",""),143:(Y,"l","differentiable DL surrogate for transport simulation"),144:(N,"h",""),
145:(Y,"h",""),146:(Y,"h",""),147:(Y,"h",""),148:(N,"h",""),149:(N,"h","symmetry reductions, no NN"),150:(N,"h",""),151:(Y,"h",""),152:(N,"h",""),
153:(Y,"h",""),154:(Y,"l","physics-informed CNN for plasma wake fields"),155:(Y,"h",""),156:(Y,"h",""),157:(N,"l",""),158:(Y,"h",""),159:(Y,"h",""),
160:(Y,"h",""),161:(N,"h",""),162:(Y,"h",""),163:(N,"h",""),164:(N,"l","learned diffusion operator for image denoising"),165:(Y,"h",""),166:(Y,"h",""),
167:(Y,"h",""),168:(Y,"h",""),169:(Y,"h",""),170:(N,"l","antenna design"),171:(N,"l",""),172:(Y,"h",""),173:(Y,"l","DL surrogate of reactive transport"),
174:(N,"h",""),175:(Y,"h",""),176:(Y,"h",""),177:(Y,"l","EIT with physics cycle consistency"),178:(Y,"h",""),179:(N,"l",""),
180:(Y,"l","diffusion models for fluid-flow prediction"),181:(N,"l","PDE-inspired architecture for images"),182:(N,"h",""),183:(Y,"l","NN-enhanced micromechanics"),
184:(Y,"h",""),185:(Y,"h",""),186:(Y,"h",""),187:(Y,"h",""),188:(Y,"h",""),189:(Y,"h",""),190:(Y,"h",""),191:(Y,"h",""),
192:(Y,"l","neural time fields solving the eikonal equation"),193:(N,"l",""),194:(Y,"l","PIML for porous-media flow properties"),195:(Y,"h",""),196:(Y,"h",""),
197:(Y,"h",""),198:(Y,"h",""),199:(N,"l",""),200:(N,"l","kernel methods, not NN"),201:(Y,"l","physics-informed RNN for crystallisation ODE"),202:(Y,"h",""),
203:(Y,"l","NN for phase-change heat transfer"),204:(Y,"h",""),205:(N,"h",""),206:(Y,"h","converter ODEs"),207:(N,"l",""),208:(N,"h",""),209:(N,"l",""),
210:(Y,"l","wavelet neural operator for elastography inversion"),211:(N,"h",""),212:(N,"l","reduced plasma model without NN"),213:(N,"l","operator inference"),
214:(N,"h",""),215:(Y,"h",""),216:(Y,"h",""),217:(Y,"h",""),218:(N,"h","retracted; fault localisation"),219:(Y,"h",""),220:(Y,"l","message-passing neural operators"),
221:(Y,"l","LM-ANN for Darcy-Forchheimer flow equations"),222:(Y,"h",""),223:(N,"l",""),224:(N,"l",""),225:(N,"l","FNO for generic time series"),
226:(Y,"l","RNN for nanofluid flow equations"),227:(Y,"h",""),228:(N,"l",""),229:(N,"l","PPV, empirical"),230:(N,"l","image-based strain estimation"),
231:(Y,"h",""),232:(Y,"h",""),233:(Y,"h",""),234:(Y,"l","NN for nanofluid flow equations"),235:(N,"h",""),236:(Y,"h",""),
237:(Y,"l","stochastic intelligence for nanofluid flow equations"),238:(Y,"h",""),239:(Y,"h",""),240:(Y,"h",""),241:(N,"h",""),242:(Y,"h",""),
243:(Y,"l","autoencoder parametrisation of incompressible flows"),244:(Y,"l","physics-informed NN busbar temperature"),245:(Y,"l","ML emulator of DNS fields"),
246:(Y,"l","ML discovery of chemotaxis PDE"),247:(Y,"h",""),248:(N,"h",""),249:(Y,"h",""),
}


def apply(name, labels):
    path = G1 / f"relevance_{name}_v3.csv"
    df = pd.read_csv(path, encoding=EXCEL_ENCODING, dtype={"in_scope": str, "confidence": str, "labeller": str, "notes": str})
    assert sorted(labels) == list(df.index), "label indices do not match the sample"
    df["in_scope"] = [labels[i][0] for i in df.index]
    df["confidence"] = ["high" if labels[i][1] == "h" else "low" for i in df.index]
    df["labeller"] = "LLM (Claude), pending human audit"
    df["notes"] = [labels[i][2] for i in df.index]
    df.to_csv(path, index=False, encoding=EXCEL_ENCODING)
    print(name, df["in_scope"].value_counts().to_dict(), "low:", int((df["confidence"] == "low").sum()))


if __name__ == "__main__":
    apply(sys.argv[1], TEST if sys.argv[1] == "test" else TRAIN)
