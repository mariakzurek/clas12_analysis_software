/*
 * processing_data_pid_training.groovy
 *
 * Author:   Maria Zurek (PI) / Cooper Bell <SULI student>
 * Created:  2026-06
 * Purpose:  Produce a per-FD-track ntuple for ML kaon/pion PID classifier training.
 *           Data equivalent of processing_mc_pid_training.groovy.
 *           Designed for CLAS12 pass-2 cooked HIPO files (real data).
 *           Applies QADB filtering. Warns if runnum == 11 (MC bleed-through).
 *
 * ─── What this script does ────────────────────────────────────────────────────
 *  1. Requires a trigger electron at REC::Particle row 0 that passes a set of
 *     primitive cuts built directly from generic_tests, pid_cuts, and fiducial_cuts
 *     — same logical cuts as the production SIDIS analysis without going through
 *     analysis_fitter.electron_test().
 *     After the electron passes, applies the Capobianco momentum correction to the
 *     electron before computing DIS variables inline from the corrected 4-vector.
 *  2. Loops over every other REC::Particle row; writes ONE ROW per FD charged
 *     hadron track with pid ∈ {+211, +321, +2212} (positive hadrons only) that
 *     also passes:
 *       (a) generic_tests.forward_detector_cut()  — |status| ∈ [2000, 4000)
 *       (b) connorHadronVertexCut()               — |vz_e - vz_hadron| < 20 cm
 *                                                   (Connor Pecar, Section IV.C.1)
 *       (c) fiducial_cuts.dc_fiducial_cut()        — DC region edge cuts
 *     NOTE: hadron vertex cut (b) is applied to ALL hadrons (including protons).
 *     This diverges from analysis_fitter.proton_test(), which has vertex_cut
 *     commented out as a known bug. Here connorHadronVertexCut is an
 *     acceptance/quality cut (rejects ghost tracks with bad vz relative to the
 *     trigger electron), not a species-specific PID cut.
 *     We do NOT replicate that bug in the training ntuple.
 *     NO chi2pid filter — chi2pid is written as a feature (col 17), not used as a cut.
 *  3. Reads per-track detector-response variables directly from HIPO banks (no
 *     analyzer Java class is used — avoids the getIndex() index-alignment bug).
 *  4. Applies QADB filtering (cook "latest", 10 defect checks including
 *     PossiblyNoBeam, allowMiscBit run list). Applies run-range exclusions for
 *     Hall-C bleedthrough and outbending RGC Sp23.
 *  5. Applies Capobianco electron momentum correction (inbending_momentum_corrections
 *     / outbending_momentum_corrections, ivec=0) for:
 *       - RGA Fa18 (runnum 4763–5666): corEl=3 (Fa18 pass-2)
 *       - RGA Sp19 (runnum 6616–6783): corEl=2 (Sp19 pass-2)
 *       - All other runs: no correction (corEl=0 → dp=0, no-op)
 *     Applied to the electron BEFORE computing DIS variables.
 *  6. Applies proton energy-loss correction (proton_energy_loss_corrections)
 *     for FD positive tracks with pid == 2212, mirroring analysis_fitter.java:317.
 *     Covers RGA Fa18 Inb/Out, RGA Sp19 Inb, RGC Su22 Inb; outside those ranges
 *     the correction is a no-op (all coefficients stay 0).
 *  7. Applies π+ Stefan energy-loss correction (stefan_piplus_energy_loss_corrections)
 *     followed by π+ Capobianco momentum correction for FD tracks with pid == 211
 *     and h_sector ∈ [1,6]. Energy-loss is applied first (same convention as proton).
 *     Stefan correction is a no-op outside Fa18 Inb/Out and Sp19 Inb run ranges.
 *     Capobianco: corPip=3 (Fa18 pass-2) for Fa18 Inb (4763–5419), corPip=2 (Sp19
 *     pass-2) for Sp19 Inb (6616–6783), corPip=0 (no correction) for Fa18 Out and
 *     all other periods (outbending momentum_corrections has no π+ table).
 *
 * ─── Electron momentum correction note ───────────────────────────────────────
 *  On branch suli_kaon_pid_main, the Capobianco electron momentum correction
 *  calls in analysis_fitter.java (lines 272–276) are COMMENTED OUT, so no
 *  existing SIDIS production script applies them. This data PID training script
 *  DOES apply them, as the PI decision for training data quality.
 *  DIS kinematics (Q2/W/x/y/nu) are computed INLINE from the corrected electron
 *  4-vector (not via the Inclusive analyzer), mirroring the LorentzVector
 *  construction pattern in Inclusive.java but using e_px/py/pz AFTER the
 *  Capobianco correction has been applied. This ensures the training ntuple
 *  DIS variables are fully consistent with the corrected electron momentum.
 *  Document this divergence from production SIDIS scripts when comparing
 *  distributions.
 *
 * ─── No MC truth ─────────────────────────────────────────────────────────────
 *  This is a data script. MC::Lund and MC::Particle banks are not loaded.
 *  The three MC truth columns (51–53 in the MC script) are omitted → 50 columns.
 *
 * ─── Missing-value convention ─────────────────────────────────────────────────
 *  Any -9999 in the output means the variable is missing or invalid for this track.
 *  Causes: bank absent, no matching pindex, no RICH hit. Downstream user must
 *  decide to impute, drop, or use missingness itself as a feature (missingness
 *  in PCAL and FTOF layer 2 is physically meaningful for PID).
 *
 * ─── Output columns (57 total) ────────────────────────────────────────────────
 *  Event-level (8):
 *   1  runnum          2  evnum           3  helicity
 *   4  Q2              5  W
 *   6  x               7  y               8  nu
 *  Per-track kinematics (7):
 *   9  pid             10 p               11 theta
 *   12 phi             13 vz              14 sector
 *   15 status
 *  Per-track missing mass (3) — appended at end, NOT ML features:
 *   52 Mx_epiX  (missing mass with pi+ hypothesis, m_h = 0.139570 GeV)
 *   53 Mx_eKX   (missing mass with K+  hypothesis, m_h = 0.493677 GeV)
 *   54 Mx_epX   (missing mass with p   hypothesis, m_h = 0.938272 GeV)
 *   Sentinel: -9999 when M_X^2 < 0 (unphysical). Masses from kinematic_variables.java.
 *   Electron 4-vector from corrected e_px/e_py/e_pz (Capobianco). Beam energy from `Eb`.
 *  Per-track z (SIDIS energy fraction, 3) — appended at end, NOT ML features:
 *   55 z_epi   (z assuming pi+ hypothesis, z = sqrt(p^2+M_pi^2)/nu)
 *   56 z_eK    (z assuming K+  hypothesis, z = sqrt(p^2+M_K^2)/nu)
 *   57 z_ep    (z assuming p   hypothesis, z = sqrt(p^2+M_p^2)/nu)
 *   Sentinel: -9999 when nu is missing/non-positive or h_p is the -9999 sentinel.
 *   z is physically bounded in [0,1] for SIDIS; phase-space tails may give z > 1 — NOT clipped.
 *  Per-track ML features — 13 features (beta + FTOF 1A/1B + ECAL inner/outer) + chi2pid + nphe_htcc + nphe_ltcc (16):
 *   16 beta            17 chi2pid
 *   18 ftof_energy_1A  19 ftof_energy_1B  20 ftof_time_1A   21 ftof_time_1B
 *   22 ftof_path_1A    23 ftof_path_1B
 *   24 ecin_energy     25 ecout_energy    26 ecin_time       27 ecout_time
 *   28 ecin_path       29 ecout_path
 *   30 nphe_htcc
 *   31 nphe_ltcc
  *   NOTE: all 5 time columns (ftof_time_1A, ftof_time_1B, ecin_time, ecout_time,
  *   pcal_time, ftof_time_2) store TIME OF FLIGHT, not absolute paddle time.
  *   The subtraction uses REC::Particle.vt for each track, which is the per-track
  *   start time (trigger-electron timing + RF correction using this particle's vz).
  *   Typical values ~25 ns. MISSING (-9999) if REC::Particle.vt is absent or invalid
  *   (EB sentinel: vt <= 0).
 *  PCAL + FTOF layer 2 — included for completeness; evaluate for training (6):
 *   32 pcal_energy     33 pcal_time       34 pcal_path
 *   35 ftof_energy_2   36 ftof_time_2     37 ftof_path_2
 *  RICH — cross-check only, NOT training features (14):
 *   38 rich_emilay     39 rich_emico      40 rich_emqua      41 rich_best_PID
 *   42 rich_RQ         43 rich_ReQ        44 rich_el_logl    45 rich_pi_logl
 *   46 rich_k_logl     47 rich_pr_logl    48 rich_best_ch    49 rich_best_c2
 *   50 rich_best_RL    51 rich_best_ntot
 *
 * ─── How to run ───────────────────────────────────────────────────────────────
 *   ./processing.csh processing_scripts/processing_data_pid_training.groovy \
 *       <hipo_dir> <output_basename> [<n_files>] [<beam_E>] [<runnum_override>]
 *
 *  Example (RGA Fa18 inbending, all files, beam energy from bank):
 *   ./processing.csh processing_scripts/processing_data_pid_training.groovy \
 *       /path/to/rga_fa18_inb/ data_pid_training 0
 *
 *  Example (force beam energy override):
 *   ./processing.csh processing_scripts/processing_data_pid_training.groovy \
 *       /path/to/rga_fa18_inb/ data_pid_training 0 10.6041
 *
 * ─── Files that need updating but are NOT changed here ────────────────────────
 *  - convert_txt_to_root.cpp  : add a new case (e.g. case 8) with 51 branches.
 *  - processing.csh           : add this script name → convert_arg3 mapping.
 */

// ─── Imports ──────────────────────────────────────────────────────────────────
import org.jlab.io.hipo.HipoDataSource
import org.jlab.io.hipo.HipoDataEvent
import org.jlab.io.hipo.HipoDataBank
import org.jlab.clas.physics.LorentzVector
import groovy.io.FileType
import clasqa.QADB
import extended_kinematic_fitters.generic_tests
import extended_kinematic_fitters.fiducial_cuts
import extended_kinematic_fitters.pid_cuts
import extended_kinematic_fitters.energy_loss_corrections
import extended_kinematic_fitters.momentum_corrections
import analyzers.kinematic_variables

// ─── Main class ───────────────────────────────────────────────────────────────
public class PIDDataTrainingScript {

    // ── Constants ──────────────────────────────────────────────────────────────
    static final double MISSING = -9999.0

    // ── PDG mass constants for missing-mass hypotheses ─────────────────────────
    // Values taken verbatim from kinematic_variables.particle_mass() (lines 29,35,37,39).
    // Any future PDG update must be propagated in lockstep with kinematic_variables.java.
    static final double M_ELECTRON = 0.0005109989461  // GeV  (kinematic_variables.java:29)
    static final double M_PIPLUS   = 0.139570          // GeV  (kinematic_variables.java:35)
    static final double M_KPLUS    = 0.493677          // GeV  (kinematic_variables.java:37)
    static final double M_PROTON   = 0.938272          // GeV  (kinematic_variables.java:39)

    // ── Stateless helper instances (hoisted to avoid per-track allocation) ─────
    static final generic_tests       GENERIC_TESTS = new generic_tests()
    static final fiducial_cuts       FIDUCIAL_CUTS = new fiducial_cuts()
    static final pid_cuts            PID_CUTS      = new pid_cuts()
    static final energy_loss_corrections ELC       = new energy_loss_corrections()
    static final momentum_corrections    MOM_CORR  = new momentum_corrections()
    static final kinematic_variables     KIN_VARS  = new kinematic_variables()

    // Allowed hadron PIDs.
    // Positive hadrons relevant for K+ analysis: π+ (signal contamination),
    // K+ (target species), p (high-momentum K+ contamination via TOF),
    // deuteron (45, rare but EB-assigned), unidentified positive (0).
    // pid=45 and pid=0 are included so that true K+ mislabeled by EB into
    // those bins are captured in the denominator of ε_refine.
    // pid=0 is additionally guarded by a charge>0 check at the hadron loop
    // (see below) to exclude negative unidentified tracks.
    // Positrons (+11) are excluded: EB never assigns pid=+11 to a true K+
    // (HTCC threshold ~9 GeV/c, wrong calorimeter sampling fraction).
    // Negatives (π-, K-, p̄) intentionally dropped — positive hadrons only.
    static final Set<Integer> HADRON_PIDS = [211, 321, 2212, 45, 0] as Set

    // ── Stage 1: banks needed for the electron filter only ────────────────────
    // Called for every event. Cheap set — avoids loading expensive banks
    // (RICH::Particle, REC::Scintillator) on events that will be rejected by
    // the electron filter anyway.
    static Map<String, HipoDataBank> loadBanksForElectronGate(HipoDataEvent event) {
        def banks = [:]
        ["REC::Particle", "REC::Calorimeter", "REC::Traj",
         "REC::Cherenkov", "RUN::config"].each { name ->
            banks[name] = event.hasBank(name) ? (HipoDataBank) event.getBank(name) : null
        }
        return banks
    }

    // ── Stage 2: per-track banks; called only AFTER electron passes ───────────
    // Loads the remaining expensive banks into the existing map in-place.
    // NOTE: MC::Lund and MC::Particle are intentionally excluded — data only.
    static void loadRemainingBanks(HipoDataEvent event, Map banks) {
        ["REC::Scintillator", "REC::Track", "REC::Event",
         "RICH::Particle"].each { name ->
            banks[name] = event.hasBank(name) ? (HipoDataBank) event.getBank(name) : null
        }
    }

    // ── Connor's electron vertex cut (Pecar analysis note, Section IV.C.1) ─────
    //   Inbending  (torus ≤ 0):  -8 cm < vz_e < +2 cm
    //   Outbending (torus > 0): -11 cm < vz_e < +1 cm
    // Single absolute window, polarity-dependent. Same for all run periods.
    // Replaces generic_tests.vertex_cut for the trigger electron.
    static boolean connorElectronVertexCut(HipoDataBank rec, HipoDataBank run) {
        if (!rec || !run) return false
        if (rec.rows() == 0) return false
        double vz_e   = rec.getFloat("vz", 0)
        float  torus  = run.getFloat("torus", 0)
        boolean inbending = (torus <= 0)
        double vz_lo = inbending ? -8.0 : -11.0
        double vz_hi = inbending ?  2.0 :   1.0
        return (vz_e > vz_lo && vz_e < vz_hi)
    }

    // ── Connor's hadron vertex cut (Pecar analysis note, Section IV.C.1) ──────
    //   |vz_e - vz_hadron| < 20 cm
    // Relative cut (hadron vertex within 20 cm of the trigger electron vertex).
    // Polarity-independent, same for all charge signs and run periods.
    // Replaces generic_tests.vertex_cut for hadrons.
    static boolean connorHadronVertexCut(int row, HipoDataBank rec) {
        if (!rec) return false
        if (rec.rows() == 0 || row >= rec.rows()) return false
        double vz_e = rec.getFloat("vz", 0)
        double vz_h = rec.getFloat("vz", row)
        return (Math.abs(vz_e - vz_h) < 20.0)
    }

    // ── Electron filter — requires pid==11 at row 0 + primitive cut composition ─
    // Cuts mirror analysis_fitter.electron_test() but are built directly from
    // generic_tests, pid_cuts, and fiducial_cuts primitives (no analysis_fitter).
    // Signatures verified against Java source:
    //   generic_tests.forward_detector_cut(int idx, HipoDataBank rec)
    //   pid_cuts.calorimeter_energy_cut(int idx, HipoDataBank cal, HipoDataBank run)
    //   pid_cuts.calorimeter_sampling_fraction_cut(int idx, double p, HipoDataBank run, HipoDataBank cal)
    //   pid_cuts.calorimeter_diagonal_cut(int idx, double p, HipoDataBank cal, HipoDataBank run)
    //   fiducial_cuts.pcal_fiducial_cut(int idx, int strictness, HipoDataBank run, HipoDataBank rec, HipoDataBank cal)
    //   fiducial_cuts.dc_fiducial_cut(int idx, HipoDataBank rec, HipoDataBank traj, HipoDataBank run)
    // Electron vertex cut: connorElectronVertexCut (replaces generic_tests.vertex_cut).
    // NOTE: calorimeter_energy_cut IS enabled here (stricter electron sample for
    // training data quality), even though analysis_fitter.electron_test has it
    // commented out on this branch.
    static boolean passElectronCuts(Map banks) {
        def rec  = banks["REC::Particle"]
        def cal  = banks["REC::Calorimeter"]
        def traj = banks["REC::Traj"]
        def run  = banks["RUN::config"]
        if (!rec || !cal || !traj || !run) return false
        if (rec.rows() == 0 || rec.getInt("pid", 0) != 11) return false
        float px = rec.getFloat("px", 0)
        float py = rec.getFloat("py", 0)
        float pz = rec.getFloat("pz", 0)
        double p_e = Math.sqrt(px*px + py*py + pz*pz)

        return p_e > 2.0 &&
               GENERIC_TESTS.forward_detector_cut(0, rec) &&
               connorElectronVertexCut(rec, run) &&
               PID_CUTS.calorimeter_energy_cut(0, cal, run) &&
               PID_CUTS.calorimeter_sampling_fraction_cut(0, p_e, run, cal) &&
               PID_CUTS.calorimeter_diagonal_cut(0, p_e, cal, run) &&
               FIDUCIAL_CUTS.pcal_fiducial_cut(0, 1, run, rec, cal) &&
               FIDUCIAL_CUTS.dc_fiducial_cut(0, rec, traj, run)
    }

    // ── Per-hadron cut filter — FD-only + vertex + DC-fiducial ───────────────
    // Built directly from generic_tests and fiducial_cuts primitives.
    // Signatures verified against Java source:
    //   generic_tests.forward_detector_cut(int idx, HipoDataBank rec)
    //   fiducial_cuts.dc_fiducial_cut(int idx, HipoDataBank rec, HipoDataBank traj, HipoDataBank run)
    // Hadron vertex cut: connorHadronVertexCut (|vz_e - vz_h| < 20 cm, replaces
    //   generic_tests.vertex_cut). Rejects ghost tracks with bad vz relative to
    //   the trigger electron. NOT a PID cut.
    static boolean passHadronCuts(int row, Map banks) {
        def rec  = banks["REC::Particle"]
        def traj = banks["REC::Traj"]
        def run  = banks["RUN::config"]
        if (!rec || !traj || !run) return false
        if (!GENERIC_TESTS.forward_detector_cut(row, rec)) return false
        if (!connorHadronVertexCut(row, rec)) return false
        return FIDUCIAL_CUTS.dc_fiducial_cut(row, rec, traj, run)
    }

    // ── FTOF extraction — REC::Scintillator, detector=12, layers 1/2/3 ───────
    // Returns [energy_1A, energy_1B, time_1A, time_1B, path_1A, path_1B, energy_2, time_2, path_2]
    // Times (indices 2, 3, 7) are time-of-flight from vertex: t_raw - track_t0.
    // If track_t0 is MISSING or the raw time itself is MISSING, the time stays MISSING.
    static double[] extractFTOF(int hadron_row, HipoDataBank scint_bank, double track_t0) {
        double[] result = [MISSING, MISSING, MISSING, MISSING, MISSING, MISSING,
                           MISSING, MISSING, MISSING]
        if (!scint_bank) return result
        for (int i = 0; i < scint_bank.rows(); i++) {
            if (scint_bank.getInt("pindex", i) != hadron_row) continue
            if (scint_bank.getInt("detector", i) != 12) continue   // FTOF detector id
            int layer = scint_bank.getInt("layer", i)
            double e = scint_bank.getFloat("energy", i)
            double t_raw = scint_bank.getFloat("time", i)
            double t = (t_raw != MISSING && track_t0 != MISSING) ? t_raw - track_t0 : MISSING
            double path = scint_bank.getFloat("path", i)
            if      (layer == 1) { result[0]=e; result[2]=t; result[4]=path }  // 1A
            else if (layer == 2) { result[1]=e; result[3]=t; result[5]=path }  // 1B
            else if (layer == 3) { result[6]=e; result[7]=t; result[8]=path }  // layer 2
        }
        return result
    }

    // ── ECAL/PCAL extraction — REC::Calorimeter, layers 1(PCAL)/4(ECin)/7(ECout)
    // Returns [pcal_e, pcal_t, pcal_path, ecin_e, ecin_t, ecin_path, ecout_e, ecout_t, ecout_path]
    // Times (indices 1, 4, 7) are time-of-flight from vertex: t_raw - track_t0.
    // If track_t0 is MISSING or the raw time itself is MISSING, the time stays MISSING.
    static double[] extractECAL(int hadron_row, HipoDataBank cal_bank, double track_t0) {
        double[] result = [MISSING, MISSING, MISSING, MISSING, MISSING, MISSING,
                           MISSING, MISSING, MISSING]
        if (!cal_bank) return result
        for (int i = 0; i < cal_bank.rows(); i++) {
            if (cal_bank.getInt("pindex", i) != hadron_row) continue
            int layer = cal_bank.getInt("layer", i)
            double e    = cal_bank.getFloat("energy", i)
            double t_raw = cal_bank.getFloat("time",  i)
            double t = (t_raw != MISSING && track_t0 != MISSING) ? t_raw - track_t0 : MISSING
            double path = cal_bank.getFloat("path",   i)
            if      (layer == 1) { result[0]=e; result[1]=t; result[2]=path }  // PCAL
            else if (layer == 4) { result[3]=e; result[4]=t; result[5]=path }  // ECin
            else if (layer == 7) { result[6]=e; result[7]=t; result[8]=path }  // ECout
        }
        return result
    }

    // ── HTCC nphe — REC::Cherenkov detector=15, sum over pindex ──────────────
    static double extractHTCC(int hadron_row, HipoDataBank cc_bank) {
        if (!cc_bank) return MISSING
        double nphe = MISSING
        for (int i = 0; i < cc_bank.rows(); i++) {
            if (cc_bank.getInt("pindex", i) != hadron_row) continue
            if (cc_bank.getInt("detector", i) != 15) continue
            double v = cc_bank.getFloat("nphe", i)
            nphe = (nphe == MISSING) ? v : nphe + v
        }
        return nphe
    }

    // ── LTCC nphe — REC::Cherenkov detector=16, sum over pindex ──────────────
    // LTCC has lower refractive index than HTCC: π fires above ~2.7 GeV/c,
    // K above ~9.6 GeV/c. Potentially useful for π/K separation in the 2.7-9.6
    // GeV band where FTOF chi2pid struggles, IF the LTCC has full coverage.
    static double extractLTCC(int hadron_row, HipoDataBank cc_bank) {
        if (!cc_bank) return MISSING
        double nphe = MISSING
        for (int i = 0; i < cc_bank.rows(); i++) {
            if (cc_bank.getInt("pindex", i) != hadron_row) continue
            if (cc_bank.getInt("detector", i) != 16) continue
            double v = cc_bank.getFloat("nphe", i)
            nphe = (nphe == MISSING) ? v : nphe + v
        }
        return nphe
    }

    // ── RICH extraction — ported from TwoParticles.java:115-138 ──────────────
    // Bank types (Hayward commit 8f7d83cf6): emilay/emico=Byte, emqua/best_PID=Short, rest=Float
    // Returns [emilay, emico, emqua, best_PID, RQ, ReQ, el_logl, pi_logl, k_logl, pr_logl,
    //          best_ch, best_c2, best_RL, best_ntot]
    static double[] extractRICH(int hadron_row, HipoDataBank rich_bank) {
        double[] r = new double[14]
        Arrays.fill(r, MISSING)
        if (!rich_bank) return r
        for (int i = 0; i < rich_bank.rows(); i++) {
            if (rich_bank.getInt("pindex", i) != hadron_row) continue
            r[0]  = rich_bank.getByte("emilay",   i)
            r[1]  = rich_bank.getByte("emico",    i)
            r[2]  = rich_bank.getShort("emqua",   i)
            r[3]  = rich_bank.getShort("best_PID",i)
            r[4]  = rich_bank.getFloat("RQ",      i)
            r[5]  = rich_bank.getFloat("ReQ",     i)
            r[6]  = rich_bank.getFloat("el_logl", i)
            r[7]  = rich_bank.getFloat("pi_logl", i)
            r[8]  = rich_bank.getFloat("k_logl",  i)
            r[9]  = rich_bank.getFloat("pr_logl", i)
            r[10] = rich_bank.getFloat("best_ch", i)
            r[11] = rich_bank.getFloat("best_c2", i)
            r[12] = rich_bank.getFloat("best_RL", i)
            r[13] = rich_bank.getFloat("best_ntot", i)
            break  // at most one RICH row per track
        }
        return r
    }

    // ── FD sector lookup — REC::Track detector=6 (DC) ─────────────────────────
    static int extractSector(int hadron_row, HipoDataBank track_bank) {
        if (!track_bank) return (int) MISSING
        for (int i = 0; i < track_bank.rows(); i++) {
            if (track_bank.getInt("pindex", i) == hadron_row &&
                track_bank.getInt("detector", i) == 6) {  // detector=6 is DC/FD track
                return track_bank.getInt("sector", i)
            }
        }
        return (int) MISSING
    }

    // ── Angle helpers ──────────────────────────────────────────────────────────
    static double phiDeg(double px, double py) {
        double phi = Math.toDegrees(Math.atan2(py, px))
        return (phi < 0.0) ? phi + 360.0 : phi
    }
    static double thetaDeg(double px, double py, double pz) {
        double r = Math.sqrt(px*px + py*py + pz*pz)
        return (r < 1e-12) ? 0.0 : Math.toDegrees(Math.acos(pz / r))
    }

    // ── Per-run beam energy lookup — mirrors BeamEnergy.java verbatim ─────────
    // Covers all run periods in BeamEnergy.java; falls back to the CLI default
    // when the run number is not in any known period (e.g., runnum == 11).
    static double beamEnergyFor(int runnum, double fallback) {
        if (runnum >= 5032  && runnum <= 5666)  return 10.6041   // RGA Fa18 Inb+Out (pass-2)
        if (runnum >= 2365  && runnum <= 2598)  return  2.22193  // RGA 2 GeV
        if (runnum >= 3030  && runnum <= 3106)  return  6.42313  // RGB 6 GeV (early)
        if (runnum >= 3819  && runnum <= 3862)  return  6.42313  // RGB 6 GeV
        if (runnum >= 3172  && runnum <= 3817)  return 10.5940   // RGA Fa18 pre-pass2
        if (runnum >= 3863  && runnum <= 4326)  return 10.5940   // RGA Fa18 pre-pass2 (cont.)
        if (runnum >= 5875  && runnum <= 6000)  return  6.535    // RGB Sp19 6 GeV
        if (runnum >= 5674  && runnum <= 5870)  return  7.546    // RGB 7.5 GeV
        if (runnum >= 6616  && runnum <= 6783)  return 10.1998   // RGA Sp19 Inb
        if (runnum >= 6120  && runnum <= 6399)  return 10.5986   // RGB Sp19 10.6
        if (runnum >= 6409  && runnum <= 6604)  return 10.1998   // RGB Sp19 10.2
        if (runnum >= 11093 && runnum <= 11283) return 10.4096   // RGC Su22 Inb
        if (runnum >= 11284 && runnum <= 11300) return  4.17179  // RGC Su22 4 GeV
        if (runnum >= 11323 && runnum <= 11571) return 10.3894   // RGC Fa22
        if (runnum >= 16042 && runnum <= 17065) return 10.5473   // RGD Fa23 (part 1)
        if (runnum >= 17067 && runnum <= 17724) return 10.5563   // RGD Fa23 (part 2)
        if (runnum >= 17725 && runnum <= 17811) return 10.5593   // RGD Fa23 (part 3)
        if (runnum >= 19249 && runnum <= 19250) return  6.39463  // RGE 6 GeV
        return fallback
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ── Main entry point ───────────────────────────────────────════════════════
    // ═══════════════════════════════════════════════════════════════════════════
    public static void main(String[] args) {

        long startTime = System.currentTimeMillis()
        final int N_COLUMNS = 57
        println("=" * 72)
        println("processing_data_pid_training.groovy  —  ML PID training ntuple (DATA)")
        println("Output: ${N_COLUMNS} columns per FD hadron track.  See header for column map.")
        println("Missing-value sentinel: ${(int)MISSING}")
        println("=" * 72)

        // ── CLI argument parsing ───────────────────────────────────────────────
        if (!args) {
            println("ERROR: Provide hipo directory as first argument.")
            System.exit(1)
        }
        def hipo_list = []
        (args[0] as File).eachFileRecurse(FileType.FILES) {
            if (it.name.endsWith('.hipo')) hipo_list << it
        }
        if (hipo_list.isEmpty()) {
            println("ERROR: No .hipo files found in ${args[0]}")
            System.exit(1)
        }

        String output_file = args.length < 2 ? "data_pid_training_out.txt" : args[1]
        File file = new File(output_file)
        file.delete()

        int n_files = (args.length < 3 ||
                       Integer.parseInt(args[2]) == 0 ||
                       Integer.parseInt(args[2]) > hipo_list.size())
                      ? hipo_list.size() : Integer.parseInt(args[2])

        double beam_energy = args.length < 4 ? 10.6041 : Double.parseDouble(args[3])
        println("Beam energy (default/override): ${beam_energy} GeV")
        println("(Run-period lookup table will override default; fallback used for runnum == 11)")

        Integer userProvidedRun = (args.length >= 5) ? Integer.parseInt(args[4]) : null

        // ── QADB setup — verbatim from processing_two_particles.groovy ─────────
        // Cook "latest", 10 defect checks including PossiblyNoBeam,
        // allowMiscBit run list for empty-target, He/ET, RICH-off, low-liveTime runs.
        QADB qa = new QADB("latest")
        qa.checkForDefect('TotalOutlier')
        qa.checkForDefect('TerminalOutlier')
        qa.checkForDefect('MarginalOutlier')
        qa.checkForDefect('SectorLoss')
        qa.checkForDefect('LowLiveTime')
        qa.checkForDefect('Misc')
        qa.checkForDefect('ChargeHigh')
        qa.checkForDefect('ChargeNegative')
        qa.checkForDefect('ChargeUnknown')
        qa.checkForDefect('PossiblyNoBeam')
        [
            5046, 5047, 5051, 5128, 5129, 5130, 5158, 5159,
            5160, 5163, 5165, 5166, 5167, 5168, 5169, 5180,
            5181, 5182, 5183, 5400, 5448, 5495, 5496, 5505,
            5567, 5610, 5617, 5621, 5623, 6736, 6737, 6738,
            6739, 6740, 6741, 6742, 6743, 6744, 6746, 6747,
            6748, 6749, 6750, 6751, 6753, 6754, 6755, 6756,
            6757,
            16194, 16089, 16185, 16308, 16184, 16307, 16309,
            16872, 16975,
            17763, 17764, 17765, 17766, 17767, 17768,
            17179, 17180, 17181, 17182, 17183, 17188, 17189,
            17252
        ].each { run -> qa.allowMiscBit(run) }

        // ── I/O batching ──────────────────────────────────────────────────────
        StringBuilder batchLines = new StringBuilder()
        int lineCount = 0
        final int BATCH_SIZE = 1000
        int num_events = 0, rows_written = 0

        // ═══════════════════════════════════════════════════════════════════════
        // ── File loop ─────────────────────────────────────────────────────────
        // ═══════════════════════════════════════════════════════════════════════
        for (int current_file = 0; current_file < n_files; current_file++) {
            println("\nOpening file ${current_file+1} of ${n_files}: ${hipo_list[current_file].name}")
            HipoDataSource reader = new HipoDataSource()
            reader.open(hipo_list[current_file])

            // ── Per-file runnum warning (lazy: fires on first readable event) ──
            boolean firstEvent = true

            // ── Event loop ────────────────────────────────────────────────────
            while (reader.hasEvent()) {
                ++num_events
                if (num_events % 500000 == 0) print("processed: ${num_events} events. ")

                HipoDataEvent event = reader.getNextEvent()

                // ── Run / event number ─────────────────────────────────────────
                def config_bank_raw = event.hasBank("RUN::config") ?
                    (HipoDataBank) event.getBank("RUN::config") : null
                if (!config_bank_raw) continue
                int runnum = userProvidedRun ?: config_bank_raw.getInt("run",  0)
                int evnum  = config_bank_raw.getInt("event", 0)

                // ── Run-range exclusions (verbatim from processing_two_particles.groovy) ──
                // Hall-C bleedthrough: terminate this file's event loop
                if (runnum > 16600 && runnum < 16700) break
                // Outbending RGC Sp23: skip
                if (runnum > 17768) continue

                // ── QADB filter ────────────────────────────────────────────────
                // runnum < 5020: pre-QADB era, bypass silently (consistent with
                //   the two_particles pattern: process_event = filter.isValid &&
                //   (runnum == 11 || runnum < 5020 || qa.pass(runnum, evnum)))
                // runnum == 11:  MC bleed-through; warn once, then bypass QA.
                boolean passQA = (runnum < 5020 || qa.pass(runnum, evnum))

                // ── Warn once per file if runnum == 11 (MC bleed-through) ──────
                if (firstEvent) {
                    firstEvent = false
                    if (runnum == 11) {
                        println("WARNING: file ${hipo_list[current_file].name} has runnum=11 " +
                                "(MC convention). This is a data script; MC input will bypass " +
                                "QADB but may produce meaningless kinematics.")
                    }
                }

                // MC (runnum==11): bypass QADB but keep processing (matches two_particles pattern)
                if (runnum == 11) passQA = true
                if (!passQA) continue

                // ── Stage 1: load only banks needed for the electron filter ────
                Map banks = loadBanksForElectronGate(event)
                banks["RUN::config"] = config_bank_raw  // already loaded above

                def rec_bank  = banks["REC::Particle"]
                def run_bank  = banks["RUN::config"]
                if (!rec_bank || !run_bank) continue

                // ── Event-level electron filter ────────────────────────────────
                // Require pid==11 at row 0 and full electron cuts (FD + SF + fiducial)
                if (!passElectronCuts(banks)) continue

                // ── Stage 2: load remaining banks (only for events that pass) ──
                loadRemainingBanks(event, banks)

                // ── Electron momentum correction (Capobianco) ─────────────────
                // Applied BEFORE computing DIS variables. Determines inbending vs
                // outbending from torus sign (torus <= 0 → inbending per CLAS12
                // convention, consistent with momentum_corrections.java:662).
                // runPeriod controls which Capobianco pass-2 table to use:
                //   corEl=3 → Fa18 pass-2 (runs 4763–5666)
                //   corEl=2 → Sp19 pass-2 (runs 6616–6783)
                //   corEl=0 → no correction (dp=0 no-op for all other runs)
                // ivec=0 selects the electron correction branch.
                float e_px = rec_bank.getFloat("px", 0)
                float e_py = rec_bank.getFloat("py", 0)
                float e_pz = rec_bank.getFloat("pz", 0)

                int e_sector_for_corr = extractSector(0, banks["REC::Track"])
                boolean isInbending = (run_bank.getFloat("torus", 0) <= 0)

                // Determine Capobianco correction version from run number
                int corEl = 0
                if (runnum >= 4763 && runnum <= 5666) {
                    corEl = 3  // Fa18 pass-2 (covers both Inb 4763–5419 and Out 5423–5666)
                } else if (runnum >= 6616 && runnum <= 6783) {
                    corEl = 2  // Sp19 pass-2
                }
                // else corEl = 0 → no correction

                if (e_sector_for_corr >= 1 && e_sector_for_corr <= 6 && corEl != 0) {
                    float[] e_momentum = [e_px, e_py, e_pz] as float[]
                    if (isInbending) {
                        MOM_CORR.inbending_momentum_corrections(
                            e_momentum, e_sector_for_corr, 0, corEl, 0, 0, 0)
                    } else {
                        MOM_CORR.outbending_momentum_corrections(
                            e_momentum, e_sector_for_corr, 0, corEl, 0, 0, 0)
                    }
                    e_px = e_momentum[0]
                    e_py = e_momentum[1]
                    e_pz = e_momentum[2]
                }

                // ── Per-event beam energy ──────────────────────────────────────
                // Uses the same run-period lookup table as BeamEnergy.java.
                // Falls back to the CLI default for runnum == 11 (MC bleed-through)
                // or any run period not in the table.
                double Eb = beamEnergyFor(runnum, beam_energy)

                // ── DIS kinematics from the CORRECTED electron ─────────────────
                // Pattern mirrors Inclusive.java lines 98–157 exactly, using
                // kinematic_variables methods verified against the Java source.
                // All LorentzVector construction uses setPxPyPzM (mass form), as
                // in Inclusive.java — NOT setPxPyPzE.
                //
                // Signatures verified in kinematic_variables.java:
                //   particle_mass(int pid)
                //   Q2(LorentzVector lv_q)           → -lv_q.mass2()
                //   nu(LorentzVector lv_beam, lv_e)  → lv_beam.e() - lv_e.e()
                //   x(double Q2, double nu)
                //   W(double Q2, double nu)
                //   y(double nu, LorentzVector lv_beam)

                // Corrected electron 4-vector (e_px/py/pz already corrected above)
                double m_e = KIN_VARS.particle_mass(11)
                LorentzVector lv_e = new LorentzVector()
                lv_e.setPxPyPzM(e_px, e_py, e_pz, m_e)

                // Beam 4-vector (electron, along z, no tilt — matches Inclusive.java:102–103)
                double pBeam = Math.sqrt(Math.max(0.0, Eb * Eb - m_e * m_e))
                LorentzVector lv_beam = new LorentzVector()
                lv_beam.setPxPyPzM(0.0, 0.0, pBeam, m_e)

                // Virtual photon: q = beam - e  (copy constructor + sub, as in Inclusive.java:128–129)
                LorentzVector lv_q = new LorentzVector(lv_beam)
                lv_q.sub(lv_e)

                // DIS variables — computed from the CORRECTED electron
                double nu = KIN_VARS.nu(lv_beam, lv_e)
                double Q2 = KIN_VARS.Q2(lv_q)
                double x  = KIN_VARS.x(Q2, nu)
                double W  = KIN_VARS.W(Q2, nu)
                double y  = KIN_VARS.y(nu, lv_beam)

                // ── Helicity — event-level; read once per event, not per hadron ──
                int helicity = banks["REC::Event"] != null ?
                    banks["REC::Event"].getByte("helicity", 0) : (int)MISSING

                // ── Hadron loop (skip row 0 which is the electron) ─────────────
                for (int row = 1; row < rec_bank.rows(); row++) {

                    int pid = rec_bank.getInt("pid", row)
                    if (!HADRON_PIDS.contains(pid)) continue
                    // For unidentified tracks (pid==0), require positive charge to
                    // exclude negative unidentified tracks from the denominator.
                    if (pid == 0 && rec_bank.getByte("charge", row) <= 0) continue

                    // ── Cut 1: FD-only + vertex + DC-fiducial ──────────────────
                    if (!passHadronCuts(row, banks)) continue

                    // ── Per-track kinematics ───────────────────────────────────
                    float h_px = rec_bank.getFloat("px", row)
                    float h_py = rec_bank.getFloat("py", row)
                    float h_pz = rec_bank.getFloat("pz", row)
                    float h_vz = rec_bank.getFloat("vz", row)
                    int h_status = rec_bank.getInt("status", row)

                    int h_sector = extractSector(row, banks["REC::Track"])

                    // ── Proton energy-loss correction ──────────────────────────
                    // Mirrors analysis_fitter.java:317 (ENABLED on this branch).
                    // Covers RGA Fa18 Inb/Out, RGA Sp19 Inb, RGC Su22 Inb;
                    // outside those ranges dp=0 (no-op).
                    // Signature: proton_energy_loss_corrections(int particle_Index,
                    //   float[] p_array, HipoDataBank rec_Bank, HipoDataBank run_Bank)
                    if (pid == 2212) {
                        float[] h_momentum = [h_px, h_py, h_pz] as float[]
                        ELC.proton_energy_loss_corrections(row, h_momentum, rec_bank, run_bank)
                        h_px = h_momentum[0]
                        h_py = h_momentum[1]
                        h_pz = h_momentum[2]
                    }

                    // ── π+ energy-loss correction (Stefan) + momentum correction (Capobianco) ──
                    // Energy-loss is applied FIRST (same convention as the proton block above),
                    // then momentum correction on the energy-loss-corrected vector.
                    //
                    // Stefan energy-loss:
                    //   Covers runPeriod 1 (Fa18 Inb 4763–5419), 2 (Fa18 Out 5423–5666),
                    //   3 (Sp19 Inb 6616–6783). dp=0 for all other run ranges (no-op).
                    //   Signature: stefan_piplus_energy_loss_corrections(int particle_Index,
                    //     float[] p_array, HipoDataBank rec_Bank, HipoDataBank run_Bank,
                    //     HipoDataBank track_Bank)
                    //
                    // Capobianco momentum correction (inbending only; no outbending π+ table):
                    //   corPip=3 → Fa18 pass-2 (Fa18 Inb, runs 4763–5419)
                    //   corPip=2 → Sp19 pass-2  (Sp19 Inb, runs 6616–6783)
                    //   corPip=0 → no correction (Fa18 Out and all other runs; outbending
                    //              momentum_corrections has no π+ table so dp=0 regardless,
                    //              but we pass corPip=0 explicitly to be safe)
                    if (pid == 211 && h_sector >= 1 && h_sector <= 6) {
                        // Step 1: Stefan energy-loss correction
                        float[] h_p_array = [h_px, h_py, h_pz] as float[]
                        ELC.stefan_piplus_energy_loss_corrections(
                            row, h_p_array, rec_bank, run_bank, banks["REC::Track"])
                        h_px = h_p_array[0]
                        h_py = h_p_array[1]
                        h_pz = h_p_array[2]

                        // Step 2: Capobianco momentum correction
                        // Determine corPip from run period (inbending pass-2 where available)
                        int corPip = 0
                        if      (runnum >= 4763 && runnum <= 5419) corPip = 3  // Fa18 Inb pass-2
                        else if (runnum >= 6616 && runnum <= 6783) corPip = 2  // Sp19 Inb pass-2
                        // Fa18 Out (5423–5666): outbending has no π+ table → corPip=0 (no-op)

                        if (corPip != 0) {
                            float[] h_p_array2 = [h_px, h_py, h_pz] as float[]
                            // isInbending already determined above for the electron correction
                            if (isInbending) {
                                MOM_CORR.inbending_momentum_corrections(
                                    h_p_array2, h_sector, 1, 0, corPip, 0, 0)
                            } else {
                                MOM_CORR.outbending_momentum_corrections(
                                    h_p_array2, h_sector, 1, 0, corPip, 0, 0)
                            }
                            h_px = h_p_array2[0]
                            h_py = h_p_array2[1]
                            h_pz = h_p_array2[2]
                        }
                    }

                    double h_p     = Math.sqrt(h_px*h_px + h_py*h_py + h_pz*h_pz)
                    double h_theta = thetaDeg(h_px, h_py, h_pz)
                    double h_phi   = phiDeg(h_px, h_py)

                    // ── REC::Particle scalar features ──────────────────────────
                    double beta     = rec_bank.getFloat("beta",    row)
                    double chi2pid  = rec_bank.getFloat("chi2pid", row)

                    // ── Per-track start time from REC::Particle.vt ─────────────
                    // REC::Particle.vt is the per-track start time, RF-corrected
                    // using this particle's vz. Use it (not REC::Event.startTime)
                    // to convert paddle times into actual time-of-flight from the
                    // vertex. If vt is the EB sentinel (-9999 or non-positive),
                    // the timing is invalid and we set the TOF columns to MISSING.
                    double h_vt = rec_bank.getFloat("vt", row)
                    double track_t0 = (h_vt > 0.0) ? h_vt : MISSING

                    // ── FTOF detector responses ────────────────────────────────
                    double[] ftof = extractFTOF(row, banks["REC::Scintillator"], track_t0)
                    // [energy_1A, energy_1B, time_1A, time_1B, path_1A, path_1B,
                    //  energy_2,  time_2,    path_2]
                    // times are TOF from vertex (~25 ns); MISSING if vt absent/invalid.

                    // ── ECAL / PCAL calorimeter responses ──────────────────────
                    double[] ecal = extractECAL(row, banks["REC::Calorimeter"], track_t0)
                    // [pcal_e, pcal_t, pcal_path, ecin_e, ecin_t, ecin_path, ecout_e, ecout_t, ecout_path]
                    // times are TOF from vertex (~25 ns); MISSING if vt absent/invalid.

                    // ── HTCC nphe ──────────────────────────────────────────────
                    double nphe_htcc = extractHTCC(row, banks["REC::Cherenkov"])
                    double nphe_ltcc = extractLTCC(row, banks["REC::Cherenkov"])

                    // ── RICH variables ─────────────────────────────────────────
                    double[] rich = extractRICH(row, banks["RICH::Particle"])

                    // ── Missing-mass hypotheses (per-track, 3 values) ──────────
                    // Formula: Mx^2 = (E_beam + M_p - E_e - E_h)^2
                    //                 - (-e_px - h_px)^2 - (-e_py - h_py)^2
                    //                 - (p_beam - e_pz - h_pz)^2
                    // Uses beam-energy variable `Eb` (per-event, computed above from run table).
                    // Uses corrected scattered electron 4-vector (e_px/e_py/e_pz from Capobianco
                    // block above; h_px/h_py/h_pz are post-correction hadron momenta).
                    // Sentinel MISSING (-9999) written when M_X^2 < 0 (unphysical).
                    double e_E_mx   = Math.sqrt(e_px*e_px + e_py*e_py + e_pz*e_pz + M_ELECTRON*M_ELECTRON)
                    double p_beam_mx = Math.sqrt(Math.max(0.0, Eb*Eb - M_ELECTRON*M_ELECTRON))
                    double miss_px_mx = -e_px - h_px
                    double miss_py_mx = -e_py - h_py
                    double miss_pz_mx =  p_beam_mx - e_pz - h_pz
                    // pi+ hypothesis
                    double E_h_pi_mx  = Math.sqrt(h_p*h_p + M_PIPLUS*M_PIPLUS)
                    double miss_E_pi_mx = Eb + M_PROTON - e_E_mx - E_h_pi_mx
                    double Mx2_pi_mx = miss_E_pi_mx*miss_E_pi_mx - miss_px_mx*miss_px_mx - miss_py_mx*miss_py_mx - miss_pz_mx*miss_pz_mx
                    double Mx_epiX = (Mx2_pi_mx >= 0.0) ? Math.sqrt(Mx2_pi_mx) : MISSING
                    // K+ hypothesis
                    double E_h_K_mx   = Math.sqrt(h_p*h_p + M_KPLUS*M_KPLUS)
                    double miss_E_K_mx  = Eb + M_PROTON - e_E_mx - E_h_K_mx
                    double Mx2_K_mx  = miss_E_K_mx*miss_E_K_mx - miss_px_mx*miss_px_mx - miss_py_mx*miss_py_mx - miss_pz_mx*miss_pz_mx
                    double Mx_eKX  = (Mx2_K_mx  >= 0.0) ? Math.sqrt(Mx2_K_mx)  : MISSING
                    // proton hypothesis
                    double E_h_p_mx   = Math.sqrt(h_p*h_p + M_PROTON*M_PROTON)
                    double miss_E_p_mx  = Eb + M_PROTON - e_E_mx - E_h_p_mx
                    double Mx2_p_mx  = miss_E_p_mx*miss_E_p_mx - miss_px_mx*miss_px_mx - miss_py_mx*miss_py_mx - miss_pz_mx*miss_pz_mx
                    double Mx_epX  = (Mx2_p_mx  >= 0.0) ? Math.sqrt(Mx2_p_mx)  : MISSING

                    // ── SIDIS energy fraction z (per-track, 3 hypotheses) ──────────
                    // z = E_h / nu, where E_h = sqrt(p^2 + m_h^2) under each mass hypothesis.
                    // Sentinel -9999 written when nu is missing, undefined, or non-positive.
                    // h_p sentinel guard: passHadronCuts requires real REC::Particle momentum,
                    // so h_p < 0 should not occur here, but check defensively.
                    double z_epi, z_eK, z_ep
                    if (nu > 0.0 && h_p > 0.0) {
                        z_epi = Math.sqrt(h_p*h_p + M_PIPLUS*M_PIPLUS) / nu
                        z_eK  = Math.sqrt(h_p*h_p + M_KPLUS*M_KPLUS)   / nu
                        z_ep  = Math.sqrt(h_p*h_p + M_PROTON*M_PROTON) / nu
                    } else {
                        z_epi = MISSING
                        z_eK  = MISSING
                        z_ep  = MISSING
                    }

                    // ── Assemble output row (57 columns; see header + final println) ──
                    StringBuilder row_sb = new StringBuilder()
                    // Event-level
                    row_sb.append(runnum).append(' ').append(evnum).append(' ').append(helicity).append(' ')
                    row_sb.append(Q2).append(' ').append(W).append(' ')
                    row_sb.append(x).append(' ').append(y).append(' ').append(nu).append(' ')
                    // Per-track kinematics
                    row_sb.append(pid).append(' ').append(h_p).append(' ').append(h_theta).append(' ')
                    row_sb.append(h_phi).append(' ').append(h_vz).append(' ').append(h_sector).append(' ')
                    row_sb.append(h_status).append(' ')
                    // ML features — 13 features (beta + FTOF 1A/1B + ECAL inner/outer) + chi2pid + nphe_htcc + nphe_ltcc
                    row_sb.append(beta).append(' ').append(chi2pid).append(' ')
                    row_sb.append(ftof[0]).append(' ').append(ftof[1]).append(' ')  // energy 1A, 1B
                    row_sb.append(ftof[2]).append(' ').append(ftof[3]).append(' ')  // time 1A, 1B
                    row_sb.append(ftof[4]).append(' ').append(ftof[5]).append(' ')  // path 1A, 1B
                    row_sb.append(ecal[3]).append(' ').append(ecal[6]).append(' ')  // ecin_e, ecout_e
                    row_sb.append(ecal[4]).append(' ').append(ecal[7]).append(' ')  // ecin_t, ecout_t
                    row_sb.append(ecal[5]).append(' ').append(ecal[8]).append(' ')  // ecin_path, ecout_path
                    row_sb.append(nphe_htcc).append(' ')
                    row_sb.append(nphe_ltcc).append(' ')
                    // PCAL + FTOF layer 2
                    row_sb.append(ecal[0]).append(' ').append(ecal[1]).append(' ').append(ecal[2]).append(' ')
                    row_sb.append(ftof[6]).append(' ').append(ftof[7]).append(' ').append(ftof[8]).append(' ')
                    // RICH (14 vars)
                    rich.each { row_sb.append(it).append(' ') }
                    // Missing-mass hypotheses (cols 52-54) — appended at end (defensive ordering)
                    row_sb.append(Mx_epiX).append(' ').append(Mx_eKX).append(' ').append(Mx_epX)
                    // Z (SIDIS energy fraction, cols 55-57) — appended after Mx hypotheses
                    row_sb.append(' ').append(z_epi).append(' ').append(z_eK).append(' ').append(z_ep)
                    row_sb.append('\n')

                    batchLines.append(row_sb)
                    lineCount++
                    rows_written++

                    if (lineCount >= BATCH_SIZE) {
                        file.append(batchLines.toString())
                        batchLines.setLength(0)
                        lineCount = 0
                    }

                } // end hadron loop

            } // end event loop

            reader.close()

            // Flush any remaining rows for this file
            if (batchLines.length() > 0) {
                file.append(batchLines.toString())
                batchLines.setLength(0)
                lineCount = 0
            }

        } // end file loop

        // ── Final flush (safety) ──────────────────────────────────────────────
        if (batchLines.length() > 0) {
            file.append(batchLines.toString())
        }

        // ── Column index map (printed at runtime) ─────────────────────────────
        println("\n" + "=" * 72)
        println("Column index map (${N_COLUMNS} columns, space-separated):")
        println(" EVENT-LEVEL:")
        println("  1:runnum  2:evnum  3:helicity")
        println("  4:Q2  5:W  6:x  7:y  8:nu")
        println(" PER-TRACK KINEMATICS:")
        println("  9:pid  10:p  11:theta  12:phi  13:vz  14:sector  15:status")
        println(" ML FEATURES (16 features: beta + FTOF 1A/1B + ECAL inner/outer; + chi2pid + nphe_htcc + nphe_ltcc):")
        println("  16:beta  17:chi2pid")
        println("  18:ftof_energy_1A  19:ftof_energy_1B  20:ftof_time_1A  21:ftof_time_1B")
        println("  22:ftof_path_1A  23:ftof_path_1B")
        println("  24:ecin_energy  25:ecout_energy  26:ecin_time  27:ecout_time")
        println("  28:ecin_path  29:ecout_path")
        println("  30:nphe_htcc")
        println("  31:nphe_ltcc")
        println(" PCAL + FTOF LAYER 2 (included for completeness; evaluate for training):")
        println("  32:pcal_energy  33:pcal_time  34:pcal_path")
        println("  35:ftof_energy_2  36:ftof_time_2  37:ftof_path_2")
        println(" RICH CROSS-CHECK (NOT training features):")
        println("  38:rich_emilay  39:rich_emico  40:rich_emqua  41:rich_best_PID")
        println("  42:rich_RQ  43:rich_ReQ")
        println("  44:rich_el_logl  45:rich_pi_logl  46:rich_k_logl  47:rich_pr_logl")
        println("  48:rich_best_ch  49:rich_best_c2  50:rich_best_RL  51:rich_best_ntot")
        println(" MISSING-MASS HYPOTHESES (appended; NOT ML features):")
        println("  52:Mx_epiX  (e+pi+X missing mass, m_h=0.139570 GeV; -9999 if Mx^2<0)")
        println("  53:Mx_eKX   (e+K+X  missing mass, m_h=0.493677 GeV; -9999 if Mx^2<0)")
        println("  54:Mx_epX   (e+p+X  missing mass, m_h=0.938272 GeV; -9999 if Mx^2<0)")
        println(" Z (SIDIS ENERGY FRACTION, appended; NOT ML features):")
        println("  55:z_epi  (z with pi+ hypothesis, sqrt(p^2+M_pi^2)/nu; -9999 if nu<=0)")
        println("  56:z_eK   (z with K+  hypothesis, sqrt(p^2+M_K^2)/nu;  -9999 if nu<=0)")
        println("  57:z_ep   (z with p   hypothesis, sqrt(p^2+M_p^2)/nu;  -9999 if nu<=0)")
        println("=" * 72)
        println("Output file: ${output_file}")
        println("Events processed: ${num_events}")
        println("Hadron rows written: ${rows_written}")
        long elapsed = System.currentTimeMillis() - startTime
        println("Elapsed time: ${elapsed} ms")

    } // end main

} // end class PIDDataTrainingScript
