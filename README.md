# PennyLane · Semillero en Computación Cuántica

Material de trabajo del **Semillero en Computación Cuántica** de la Universidad del Rosario
para aprender computación cuántica con [PennyLane](https://pennylane.ai), la librería de
Xanadu para programación cuántica diferenciable.

## 📓 Notebooks

| # | Notebook | Tema | Abrir |
|---|---|---|---|
| 01 | [`01_circuitos_iqp.ipynb`](notebooks/01_circuitos_iqp.ipynb) | Circuitos IQP: estimación clásica Monte Carlo, simulación exacta, descomposición, muestreo y entrenamiento con JAX + Optax | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/semillero-computacion-cuantica-urosario/PennyLane/blob/main/notebooks/01_circuitos_iqp.ipynb) |

### 01 · Circuitos IQP

Los circuitos IQP (*Instantaneous Quantum Polynomial-time*) tienen la forma
U = H⊗ⁿ D H⊗ⁿ, con D diagonal. Son difíciles de muestrear clásicamente, pero algunos de
sus valores esperados se pueden estimar de forma clásica y eficiente. En este notebook:

- Se definen las compuertas del circuito como subconjuntos de qubits.
- Se estiman valores esperados ⟨ZᵢZⱼ⟩ de forma clásica con `qp.qnn.iqp_expval`.
- Se compara con la simulación exacta en `lightning.qubit` usando la plantilla `qp.IQP`.
- Se descompone el circuito en compuertas nativas (`Hadamard` y `MultiRZ`).
- Se toman muestras (*shots*) del circuito.
- Se entrenan los parámetros con JAX y Optax como si fuera una red neuronal.

**Crédito:** reconstruido a partir del live-coding de la sesión *IQP* del
**WISER Global Quantum Program** (junio de 2026), presentada por Catalina Albornoz.

## ⚙️ Requisitos

- Python 3.10 o superior
- `pennylane >= 0.45.0`
- `jax`, `optax`, `matplotlib`

```bash
pip install "pennylane>=0.45.0" jax optax matplotlib
```

> ⚠️ Algunas funciones usadas (`qp.IQP`, `qp.qnn.iqp_expval`, `qp.gate_sets`,
> `pennylane.labs`) son muy recientes o experimentales. Con versiones anteriores de
> PennyLane pueden aparecer errores `AttributeError`, y su API puede cambiar entre versiones.

La forma más fácil de empezar es abrir el notebook en Google Colab con el botón de la tabla.

## 🤝 Cómo contribuir

1. Crea una rama (o un *fork*) a partir de `main`.
2. Agrega o mejora un notebook siguiendo la convención `NN_tema.ipynb` dentro de `notebooks/`.
3. Antes de hacer *commit*, reinicia el kernel y ejecuta todo para verificar que funciona.
4. Abre un *pull request* explicando qué cambiaste y por qué.

## 📚 Recursos

- [Documentación de PennyLane](https://docs.pennylane.ai)
- [Demos de PennyLane](https://pennylane.ai/qml/demonstrations)
- Bremner, Jozsa y Shepherd (2011), *Classical simulation of commuting quantum computations
  implies collapse of the polynomial hierarchy*, Proc. R. Soc. A 467, 459–472.
