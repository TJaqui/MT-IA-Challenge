# MT inversion

This folder contains a physics-guided neural inversion workflow for the local
`database/*.npz` files and `DescripcionRETO.pdf`.

The implemented training loss follows the PDF equation (4):

```text
L = supervised model fidelity + physics data fidelity + regularization
```

- Supervised term: MSE between true and predicted `log10(resistivities)`.
- Physics term: compares observed `zxy` with `F(G(d))` using log amplitude and
  phase. The direct operator `F` is the standard 1D MT recursive impedance
  formula and was verified against stored `zxy`.
- Regularization term: selected per attempt.

The only parameters changed between the predefined attempts are:

- neural-network architecture
- regularizer type and weight

The data generator, stored database, frequency grid, thickness vector and direct
operator are not modified.

Run the full predefined sweep:

```powershell
python inversion/run_attempts.py --train-count 12000 --val-count 2000 --epochs 12
```

Run one attempt:

```powershell
python inversion/train_inversion.py --attempt a1_mlp_smooth_l2
```

Evaluate a trained network:

```powershell
python inversion/evaluate_inversion.py --weights outputs/a1_mlp_smooth_l2/best_model.pt --input database/model_0000000.npz
```
