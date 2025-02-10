import numpy as np

class ODESolver:
    def __init__(self,f, event_func=None):
        # Wrap user's f in a new function that always
        # converts list/tuple to array (or let array be array)
        self.f = lambda u,t: np.asarray(f(u,t), float)
        self.event_func = event_func

    def advance(self):
        """Advance solution one time step."""
        raise NotImplementedError      # implement in subclass

    def set_initial_condition(self,U0):
        if isinstance(U0, (float,int)): # scalar ODE
            self.neq = 1                # No. of equations
            U0 = float(U0)
        else:                           # system of ODEs
            U0 = np.asarray(U0)
            self.neq = U0.size          # No. of equations
        self.U0 = U0

    def solve(self, time_points):
        self.t = np.asarray(time_points)
        N = len(self.t)
        if self.neq == 1:                # scalar ODE
            self.u = np.zeros(N)
        else:
            self.u = np.zeros((N,self.neq))

        # Assume that self.t[0] corresponds to self.U0
        self.u[0] = self.U0
        event_t = None
        event_u = None

        # Time loop
        for n in range(N-1):
            self.n = n
            self.u[n+1] = self.advance()
            if self.event_func and self.event_func(self.u[n+1]):
                event_t = self.t[n+1]
                event_u = self.u[n+1]
                print(f"Event triggered at t={event_t}, u={event_u}")
                return self.u[:n+2], self.t[:n+2], event_t, event_u
        return self.u, self.t, event_t, event_u

class Logistic:
    def __init__(self, alpha, R,U0):
        self.alpha, self.R, self.U0 = alpha, float(R), U0

    def __call__(self, u, t):     # f(u,t)
        return self.alpha*u*(1-u/self.R)

class ForwardEuler(ODESolver):
    def advance(self):
        u, f, n, t = self.u, self.f, self.n, self.t

        dt = t[n+1] - t[n]
        unew = u[n] + dt*f(u[n],t[n])
        return unew

class ExplicitMidpoint(ODESolver):
    def advance(self):
        """Advance the solution one time step."""
        # Create local variables tp get rid of 'self.' in
        # the numerical formula
        u, f, n, t = self.u, self.f, self.n, self.t
        # dt is not necessarily constant:
        dt = t[n+1] - t[n]
        dt2 = dt/2.0
        k1 = f(u[n], t)
        k2 = f(u[n] + dt2*k1, t[n] + dt2)
        unew = u[n] +dt*k2
        return unew

class RungeKutta4(ODESolver):
    def advance(self):
        u, f, n, t = self.u, self.f, self.n, self.t

        dt = t[n+1] - t[n]
        dt2 = dt/2.0
        k1 = f(u[n], t)
        k2 = f(u[n] +dt2*k1, t[n] +dt2)
        k3 = f(u[n] +dt2*k2, t[n] +dt2)
        k4 = f(u[n] +dt*k3, t[n] +dt)
        unew = u[n] +(dt/6.0)*(k1 +2*k2 +2*k3 +k4)
        return unew
