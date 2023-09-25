# Discovering deficiencies of neural networks
## Goal
- Find adversarial examples which are added human imperceptible noise.
  - Restriction on L<sub>2</sub> norm : 8/255
  - Restriction on L<sub>inf</sub> norm : 0.1

## Motivation & Introduction
Thanks to the outstanding performance and stable capability, deep neural networks are now widely used in various areas, e.g., image classification, object detection, malware detection and behavior classification.

However in recent, deep neural networks are found to be sensitive to adversarial perturbations. Deep neural networks output different results when pertubed images get into models. It can cause severe problems in security sensitive domain like finance, medical diagnosis and self-driving cars. So it is important to discover deficiencies of deep neural networks to make more robust models.

Since the noise is human imperceptible, there is a restriction on noise. If we measure the magnitude of noise based on L<sub>p</sub> norm, we can take upper bound to the norm.

<table width="100%">
 <td align="center">
  <img alt="image" src="https://github.com/seungguJ/NNproject_KU/assets/127372942/9b1e80e2-a176-49af-abcd-9be49d8d911c"/>
  <p>Original image</p>
 </td>
 <td align="center">
  <img alt="image" src="https://github.com/seungguJ/NNproject_KU/assets/127372942/e34b9567-36bb-4a22-9c0d-ab0b8eae90f6"/>
  <p>Adversarial image</p>
 </td>
</table>

## Dataset description
