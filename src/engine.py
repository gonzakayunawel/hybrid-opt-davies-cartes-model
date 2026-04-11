import torch
import numpy as np

class DaviesModel:
    def __init__(self, dij, Ii, Zj, device=None, Nt=500, Ntt=10):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dij = torch.from_numpy(dij).to(self.device).float()
        self.Ii = torch.from_numpy(Ii).to(self.device).float()
        self.Zj = torch.from_numpy(Zj).to(self.device).float()

        # Constants
        self.nlat = 78
        self.nlon = 71
        self.nz = 500
        self.Lr = 6
        self.Lp = 12
        self.Nt = Nt
        self.Ntt = Ntt
        self.dt = 0.1
        self.eta = 0.005
        self.tau = 0.75
        self.Ptotal = 500.0

    def run_simulation(self, beta_r, gamma_r, alpha_p, gamma_p):
        """
        Runs the Davies model simulation.

        Args:
            beta_r (float): Distance decay parameter for rioters.
            gamma_r (float): Deterrent effect of police presence.
            alpha_p (float): Attractiveness of a location to police.
            gamma_p (float): Impact of the number of rioters on the police requirement.

        Returns:
            np.ndarray: The number of rioters in each zone (Rj) at the end of the simulation.
        """
        # Initialize tensors
        Rj_t = torch.zeros(self.nz, device=self.device, dtype=torch.float32)
        Ai_t = torch.zeros((self.nlat, self.nlon), device=self.device, dtype=torch.float32)
        Pj_t = torch.zeros(self.nz, device=self.device, dtype=torch.float32)
        fjdel_t = torch.zeros((self.nz, self.Lr), device=self.device, dtype=torch.float32)
        rho_t = torch.ones((self.nlat, self.nlon), device=self.device, dtype=torch.float32)
        Ddel_t = torch.zeros((self.nz, self.Lp), device=self.device, dtype=torch.float32)
        Dj_t = torch.zeros(self.nz, device=self.device, dtype=torch.float32)

        # Local copy of Ii to avoid modifying the instance variable
        Ii_t = self.Ii.clone()

        # Precompute dij term
        # Note: self.dij is already on device
        auxij1 = torch.exp(-beta_r * self.dij)
        target_values = self.Zj[:, 2]
        dij_t = 1.0 * target_values * auxij1 / torch.max(target_values)

        counter = 0

        for nn in range(self.Nt):
            # Inner loop logic extracted from notebook
            # The notebook has nested loops: for nn in range(Nt): for mm in range(Ntt):
            # But inside the inner loop, it updates everything.
            for mm in range(self.Ntt):
                # Calculate fj_t (Attractiveness modification by police)
                fj_t = torch.exp(-torch.floor(gamma_r * Pj_t / (Rj_t + 1.0e-20)))

                # Broadcasting for attractiveness term ij
                Wij_t = fj_t * dij_t

                # Attractiveness term i
                Wi_t = Wij_t.sum(dim=2)

                # P_off computation

                # Delayed term computation for rioters
                idxr = counter - (self.Lr) * int(counter / self.Lr)
                fjdel_t[:, idxr] = fj_t
                dnm = self.Lr if counter >= self.Lr else counter + 1
                We_ij_t = fjdel_t.sum(dim=1) * dij_t / dnm
                # fjdel_t.sum(dim=1) is (500). dij_t is (78, 71, 500).
                # (500) * (78, 71, 500) -> Broadcasts on last dim.
                # We_ij_t is (78, 71, 500).

                # Flow computation
                auxw = Ai_t / (We_ij_t.sum(dim=2) + 1.0e-20)
                # Ai_t is (78, 71). We_ij_t.sum(dim=2) is (78, 71).
                # auxw is (78, 71).

                Sij_t = auxw.unsqueeze(2) * We_ij_t
                # auxw.unsqueeze(2) is (78, 71, 1). We_ij_t is (78, 71, 500).
                # Sij_t is (78, 71, 500).

                # Rioter computation
                Rj_t = torch.sum(Sij_t.sum(dim=1), dim=0)
                # Sij_t.sum(dim=1) is (78, 500).
                # Then sum(dim=0) is (500).

                # Police Interaction
                Dj_t[:] = target_values ** (alpha_p) * torch.exp(gamma_p * Rj_t[:])

                # Delayed term computation for police
                idxp = counter - (self.Lp) * int(counter / self.Lp)
                Ddel_t[:, idxp] = Dj_t[:]
                dnm = self.Lp if counter >= self.Lp else counter + 1
                Dej_t = torch.sum(Ddel_t, dim=1) / dnm

                # Police allocation
                Pj_t = self.Ptotal * Dej_t / Dej_t.sum()

                counter += 1

                # Capture rate
                fj_t = 1.0 - torch.exp(-torch.floor(Pj_t / (Rj_t + 1.0e-20)))
                Ci_t = self.tau * torch.sum(Sij_t * fj_t, dim=2)
                # fj_t is (500). Sij_t is (78, 71, 500).
                # Sij_t * fj_t -> (78, 71, 500).
                # sum(dim=2) -> (78, 71).

                # Time step for Ai and Ii
                P_off_t = rho_t * Wi_t / (1.0 + Wi_t)

                # Initial fj_t (Attractiveness):
                fj_attr = torch.exp(-torch.floor(gamma_r * Pj_t / (Rj_t + 1.0e-20)))

                Wij_t = fj_attr * dij_t
                Wi_t = Wij_t.sum(dim=2)
                P_off_t = rho_t * Wi_t / (1.0 + Wi_t)

                idxr = counter - (self.Lr) * int(counter / self.Lr)
                fjdel_t[:, idxr] = fj_attr # Storing attractiveness factor
                dnm = self.Lr if counter >= self.Lr else counter + 1

                We_ij_t = fjdel_t.sum(dim=1) * dij_t / dnm

                auxw = Ai_t / (We_ij_t.sum(dim=2) + 1.0e-20)
                Sij_t = auxw.unsqueeze(2) * We_ij_t

                Rj_t = torch.sum(Sij_t.sum(dim=1), dim=0)

                Dj_t[:] = target_values ** (alpha_p) * torch.exp(gamma_p * Rj_t[:])

                idxp = counter - (self.Lp) * int(counter / self.Lp)
                Ddel_t[:, idxp] = Dj_t[:]
                dnm = self.Lp if counter >= self.Lp else counter + 1
                Dej_t = torch.sum(Ddel_t, dim=1) / dnm

                Pj_t = self.Ptotal * Dej_t / Dej_t.sum()

                counter += 1

                # Second fj_t (Capture):
                fj_cap = 1.0 - torch.exp(-torch.floor(Pj_t / (Rj_t + 1.0e-20)))
                Ci_t = self.tau * torch.sum(Sij_t * fj_cap, dim=2)

                Ai_t += self.dt * (self.eta * P_off_t * Ii_t - Ci_t)
                Ii_t += -self.dt * self.eta * P_off_t * Ii_t

        return Rj_t.cpu().numpy()
