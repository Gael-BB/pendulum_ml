# Introduction
The aim of this project is to use deep q learning to teach a cart to balance a pendulum by accelerating back and forward. As it will be one of my 3rd year projects at the end of the year, I decided to get started early by experimenting on my own.

# Description
Inverted pendulum standing on a cart. Pendulum has mass $m$, length $l$, and angle to the downward vertical $\theta$. Cart has mass $M$, and horizontal displacement $x$.

# Calculations
For a holonomic system with generalised coordinates $q_i$:
$$
\frac{d}{dt} \left[ \frac{\partial T}{\partial \dot{q}_i} \right] - \frac{\partial T}{\partial q_i} + \frac{\partial V}{\partial q_i} = Q_i
$$
where $T$ is the total kinetic energy, $V$ is the total potential energy, and $Q_i$ are the nonconservative generalised forces.
$$
V = (1 - \cos{\theta}) m g \frac{l}{2}
$$
$$
T = \frac{1}{2} M \dot{x}^2 + \frac{1}{2} \cdot \frac{1}{12} m l^2 \dot{\theta^2} + \frac{1}{2} m ((\dot{x}+\dot{\theta}\frac{l}{2}\cos{\theta})^2 + (\dot{\theta}\frac{l}{2}\sin{\theta})^2)
$$
By substituting and simplifying, you can obtain the following equations of motions.
$$
(M + m) \ddot{x} + m l \cos{\theta} \ddot{\theta} - m l \sin{\theta} \dot{\theta}^2 = 0
$$
$$
m l \cos{\theta} \ddot{x} + \frac{1}{3} m l^2 \ddot{\theta} + m g \frac{l}{2} \sin{\theta} = 0
$$
We are interested in $\ddot{\theta}$ as a function of other parameters for our physics engine. We will ignore the effect the pendulum has on the cart.
$$
\boxed{\ddot{\theta} = -\frac{3}{l} \cos{\theta} \ddot{x} - \frac{3g}{2l} \sin{\theta}}
$$