/*
 * processing_mc_pid_training.groovy
 *
 * Author:   Maria Zurek (PI) / Cooper Bell <SULI student>
 * Created:  2026-05
 * Purpose:  Produce a per-FD-track ntuple for ML kaon/pion PID classifier training.
 *           Designed primarily for clasdis MC HIPO files (runnum == 11) but also
 *           usable on real RGA pass-2 data (QA database applied automatically).
 *
 * ─── What this script does ────────────────────────────────────────────────────
 *  1. Requires a trigger electron at REC::Particle row 0 that passes a set of
 *     primitive cuts built directly from generic_tests, pid_cuts, and fiducial_cuts
 *     — same logical cuts as the production SIDIS analysis without going through
 *     analysis_fitter.electron_test().
 *  2. Loops over every other REC::Particle row; writes ONE ROW per FD charged
 *     hadron track with pid ∈ {+211, +321, +2212} (positive hadrons only) that
 *     also passes:
 *       (a) generic_tests.forward_detector_cut()  — |status| ∈ [2000, 4000)
 *       (b) generic_tests.vertex_cut()             — vz window per run period
 *       (c) fiducial_cuts.dc_fiducial_cut()        — DC region edge cuts
 *     NO chi2pid filter — chi2pid is written as a feature (col 17), not used as a cut.
 *  3. Reads per-track detector-response variables directly from HIPO banks (no
 *     analyzer Java class is used — avoids the getIndex() index-alignment bug).
 *  4. Geometrically matches each reconstructed hadron using an anchor-then-lookup
 *     algorithm (anchor on MC::Particle, then look up parent in MC::Lund):
 *       Step A (MC::Particle) → geometric match → mc_matching_pid + mc_match_quality
 *       Step B (MC::Lund)     → PID+momentum lookup → mc_parent_pid
 *     Window: |Δφ| < 9° AND |Δθ| < 3° (MC_SCALE=3.0, scale*3° and scale*1°).
 *
 * ─── Missing-value convention ─────────────────────────────────────────────────
 *  Any -9999 in the output means the variable is missing or invalid for this track.
 *  Causes: bank absent, no matching pindex, no RICH hit, no MC match, or primary
 *  particle with no parent. Downstream user must decide to impute, drop, or use
 *  missingness itself as a feature (missingness in PCAL and FTOF layer 2 is
 *  physically meaningful for PID).
 *
 * ─── Output columns (53 total) ────────────────────────────────────────────────
 *  Event-level (8):
 *   1  runnum          2  evnum           3  helicity
 *   4  Q2              5  W
 *   6  x               7  y               8  nu
 *  Per-track kinematics (7):
 *   9  pid             10 p               11 theta
 *   12 phi             13 vz              14 sector
 *   15 status
 *  Per-track ML features — 13 features (beta + FTOF 1A/1B + ECAL inner/outer) + chi2pid + nphe_htcc (15):
 *   16 beta            17 chi2pid
 *   18 ftof_energy_1A  19 ftof_energy_1B  20 ftof_time_1A   21 ftof_time_1B
 *   22 ftof_path_1A    23 ftof_path_1B
 *   24 ecin_energy     25 ecout_energy    26 ecin_time       27 ecout_time
 *   28 ecin_path       29 ecout_path
 *   30 nphe_htcc
 *  PCAL + FTOF layer 2 — included for completeness; evaluate for training (6):
 *   31 pcal_energy     32 pcal_time       33 pcal_path
 *   34 ftof_energy_2   35 ftof_time_2     36 ftof_path_2
 *  RICH — cross-check only, NOT training features (14):
 *   37 rich_emilay     38 rich_emico      39 rich_emqua      40 rich_best_PID
 *   41 rich_RQ         42 rich_ReQ        43 rich_el_logl    44 rich_pi_logl
 *   45 rich_k_logl     46 rich_pr_logl    47 rich_best_ch    48 rich_best_c2
 *   49 rich_best_RL    50 rich_best_ntot
 *  MC truth (3):
 *   51 mc_matching_pid   52 mc_parent_pid   53 mc_match_quality
 *
 * ─── How to run ───────────────────────────────────────────────────────────────
 *   ./processing.csh processing_scripts/processing_mc_pid_training.groovy \
 *       <hipo_dir> <output_basename> [<n_files>] [<beam_E>] [<runnum_override>]
 *
 *  Example (clasdis MC, all files, 10.6041 GeV beam, force runnum=11):
 *   ./processing.csh processing_scripts/processing_mc_pid_training.groovy \
 *       /path/to/clasdis/fa18_inb/ pid_training 0 10.6041 11
 *
 * ─── Files that need updating but are NOT changed here ────────────────────────
 *  - convert_txt_to_root.cpp  : add a new case (e.g. case 7, is_mc=1) with 53 branches.
 *    The old case 7 (if it exists) is for a different 123-column ntuple and is now wrong.
 *  - processing.csh           : add this script name → convert_arg3 mapping.
 */

// ─── Imports ──────────────────────────────────────────────────────────────────
import org.jlab.io.hipo.HipoDataSource
import org.jlab.io.hipo.HipoDataEvent
import org.jlab.io.hipo.HipoDataBank
import org.jlab.clas.physics.LorentzVector
import org.jlab.clas.physics.PhysicsEvent
import groovy.io.FileType
import clasqa.QADB

import extended_kinematic_fitters.analysis_fitter
import extended_kinematic_fitters.generic_tests
import extended_kinematic_fitters.fiducial_cuts
import extended_kinematic_fitters.pid_cuts
import analyzers.BeamEnergy
import analyzers.Inclusive

// ─── Main class ───────────────────────────────────────────────────────────────
public class PIDTrainingScript {

    // ── Constants ──────────────────────────────────────────────────────────────
    static final double MISSING = -9999.0
    // MC truth matching scale factor — matches processing_mc_three_particles.groovy convention
    static final double MC_SCALE = 3.0      // phi window = scale*3°, theta window = scale*1°
    // Allowed hadron PIDs
    // Positive hadrons relevant for K+ analysis: π+ (signal contamination),
    // K+ (target species), p (high-momentum K+ contamination via TOF).
    // Negatives (π-, K-, p̄) intentionally dropped — Cooper's K+ analysis
    // uses positive hadrons only. Negative-charge training is future work.
    static final Set<Integer> HADRON_PIDS = [211, 321, 2212] as Set

    // ── Bank loading — returns map of name→bank; null if bank absent ──────────
    static Map<String, HipoDataBank> loadBanks(HipoDataEvent event) {
        def banks = [:]
        ["REC::Particle", "REC::Calorimeter", "REC::Scintillator",
         "REC::Cherenkov", "REC::Track", "REC::Traj",
         "REC::Event", "RUN::config", "MC::Lund", "MC::Particle", "RICH::Particle"].each { name ->
            banks[name] = event.hasBank(name) ? (HipoDataBank) event.getBank(name) : null
        }
        return banks
    }

    // ── Electron filter — requires pid==11 at row 0 + primitive cut composition ─
    // Cuts mirror analysis_fitter.electron_test() but are built directly from
    // generic_tests, pid_cuts, and fiducial_cuts primitives (no analysis_fitter).
    // Signatures verified against Java source:
    //   generic_tests.forward_detector_cut(int idx, HipoDataBank rec)
    //   generic_tests.vertex_cut(int idx, HipoDataBank rec, HipoDataBank run)
    //   pid_cuts.calorimeter_energy_cut(int idx, HipoDataBank cal, HipoDataBank run)
    //   pid_cuts.calorimeter_sampling_fraction_cut(int idx, double p, HipoDataBank run, HipoDataBank cal)
    //   pid_cuts.calorimeter_diagonal_cut(int idx, double p, HipoDataBank cal, HipoDataBank run)
    //   fiducial_cuts.pcal_fiducial_cut(int idx, int strictness, HipoDataBank run, HipoDataBank rec, HipoDataBank cal)
    //   fiducial_cuts.dc_fiducial_cut(int idx, HipoDataBank rec, HipoDataBank traj, HipoDataBank run)
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

        generic_tests gt = new generic_tests()
        pid_cuts pc = new pid_cuts()
        fiducial_cuts fc = new fiducial_cuts()

        return p_e > 2.0 &&
               gt.forward_detector_cut(0, rec) &&
               gt.vertex_cut(0, rec, run) &&
               pc.calorimeter_energy_cut(0, cal, run) &&
               pc.calorimeter_sampling_fraction_cut(0, p_e, run, cal) &&
               pc.calorimeter_diagonal_cut(0, p_e, cal, run) &&
               fc.pcal_fiducial_cut(0, 1, run, rec, cal) &&
               fc.dc_fiducial_cut(0, rec, traj, run)
    }

    // ── Per-hadron cut filter — FD-only + vertex + DC-fiducial ───────────────
    // Built directly from generic_tests and fiducial_cuts primitives.
    // Signatures verified against Java source:
    //   generic_tests.forward_detector_cut(int idx, HipoDataBank rec)
    //   generic_tests.vertex_cut(int idx, HipoDataBank rec, HipoDataBank run)
    //   fiducial_cuts.dc_fiducial_cut(int idx, HipoDataBank rec, HipoDataBank traj, HipoDataBank run)
    // Vertex cut is an acceptance/quality cut (rejects ghost tracks with bad vz),
    // NOT a PID cut. Without it, ghost tracks with |vz|>10 cm leak into the ntuple
    // (verified empirically: removes ~3% of EB-π+ that are unmatched to any MC particle).
    static boolean passHadronCuts(int row, int pid, Map banks) {
        def rec  = banks["REC::Particle"]
        def traj = banks["REC::Traj"]
        def run  = banks["RUN::config"]
        if (!rec || !traj || !run) return false
        generic_tests gt = new generic_tests()
        fiducial_cuts fc = new fiducial_cuts()
        if (!gt.forward_detector_cut(row, rec)) return false
        if (!gt.vertex_cut(row, rec, run)) return false
        return fc.dc_fiducial_cut(row, rec, traj, run)
    }

    // ── FTOF extraction — REC::Scintillator, detector=12, layers 1/2/3 ───────
    // Returns [energy_1A, energy_1B, time_1A, time_1B, path_1A, path_1B, energy_2, time_2, path_2]
    static double[] extractFTOF(int hadron_row, HipoDataBank scint_bank) {
        double[] result = [MISSING, MISSING, MISSING, MISSING, MISSING, MISSING,
                           MISSING, MISSING, MISSING]
        if (!scint_bank) return result
        for (int i = 0; i < scint_bank.rows(); i++) {
            if (scint_bank.getInt("pindex", i) != hadron_row) continue
            if (scint_bank.getInt("detector", i) != 12) continue   // FTOF detector id
            int layer = scint_bank.getInt("layer", i)
            double e = scint_bank.getFloat("energy", i)
            double t = scint_bank.getFloat("time",   i)
            double path = scint_bank.getFloat("path", i)
            if      (layer == 1) { result[0]=e; result[2]=t; result[4]=path }  // 1A
            else if (layer == 2) { result[1]=e; result[3]=t; result[5]=path }  // 1B
            else if (layer == 3) { result[6]=e; result[7]=t; result[8]=path }  // layer 2
        }
        return result
    }

    // ── ECAL/PCAL extraction — REC::Calorimeter, layers 1(PCAL)/4(ECin)/7(ECout)
    // Returns [pcal_e, pcal_t, pcal_path, ecin_e, ecin_t, ecin_path, ecout_e, ecout_t, ecout_path]
    static double[] extractECAL(int hadron_row, HipoDataBank cal_bank) {
        double[] result = [MISSING, MISSING, MISSING, MISSING, MISSING, MISSING,
                           MISSING, MISSING, MISSING]
        if (!cal_bank) return result
        for (int i = 0; i < cal_bank.rows(); i++) {
            if (cal_bank.getInt("pindex", i) != hadron_row) continue
            int layer = cal_bank.getInt("layer", i)
            double e    = cal_bank.getFloat("energy", i)
            double t    = cal_bank.getFloat("time",   i)
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

    // ── phi/theta helpers — identical to processing_mc_three_particles.groovy ─
    static double phi_calculation(double x, double y) {
        double phi = Math.toDegrees(Math.atan2(x, y))
        phi = phi - 90
        if (phi < 0) {
            phi = 360 + phi
        }
        phi = 360 - phi
        return phi
    }

    static double theta_calculation(double x, double y, double z) {
        double r = Math.pow(Math.pow(x,2)+Math.pow(y,2)+Math.pow(z,2), 0.5)
        return (double) (180/Math.PI)*Math.acos(z/r)
    }

    // ── MC truth matching — anchor-then-lookup algorithm ─────────────────────
    //
    // Anchors on MC::Particle (final-state only by construction). Then looks up
    // the Lund row by PID+momentum equality (MC::Particle is a subset of Lund
    // type==1).
    //
    // Step A — anchor on MC::Particle: geometric match REC track to MC::Particle
    //   (single pass). Window: |Δφ| < MC_SCALE*3° AND |Δθ| < MC_SCALE*1°.
    //   First match wins. Records matching_pid, matched_px/py/pz, and
    //   mc_match_quality = sqrt(Δφ² + Δθ²) in degrees.
    //
    // Step B — lookup parent in MC::Lund by PID + momentum: IF Step A matched,
    //   loops over MC::Lund rows. For each row: reads type as
    //   (lundBank.getByte("type", i) & 0xFF); skips if != 1. Skips if pid
    //   doesn't match matching_pid. Skips unless each momentum component matches
    //   matched_px/py/pz within 1e-4 GeV. First qualifying row is THE Lund row
    //   for this track. Reads parent_idx = (getByte("parent", i) & 0xFF) - 1.
    //   If parent_idx in [0, rows()): mc_parent_pid = Lund pid at parent_idx.
    //   Else: mc_parent_pid = -9999.
    //
    // Step C — return values:
    //   No Step A match  → [-9999, -9999, -9999].
    //   Step A matched, Step B found no Lund row → [matching_pid, -9999, mc_match_quality].
    //   Both steps matched → [matching_pid, mc_parent_pid, mc_match_quality].
    //
    // Returns [mc_matching_pid, mc_parent_pid, mc_match_quality(deg)].
    // Matching window: |Δφ| < MC_SCALE*3° AND |Δθ| < MC_SCALE*1°  (9° and 3°).
    static double[] extractMCTruth(double h_px, double h_py, double h_pz,
                                   HipoDataBank lundBank, HipoDataBank mcBank) {
        double[] result = [MISSING, MISSING, MISSING]

        double exp_phi   = phi_calculation(h_px, h_py)
        double exp_theta = theta_calculation(h_px, h_py, h_pz)

        // ── Step A: anchor on MC::Particle — geometric match for truth PID ────
        // MC::Particle contains only final-state particles (no quarks/diquarks/
        // strings), so this gives the correct truth PID for the ML label.
        boolean matched = false
        int matching_pid = 0
        double matched_px = 0.0, matched_py = 0.0, matched_pz = 0.0
        double match_dphi = 0.0, match_dtheta = 0.0

        if (mcBank) {
            for (int i = 0; i < mcBank.rows(); i++) {
                if (matched) { continue }
                double mc_px = mcBank.getFloat("px", i)
                double mc_py = mcBank.getFloat("py", i)
                double mc_pz = mcBank.getFloat("pz", i)

                double mc_phi   = phi_calculation(mc_px, mc_py)
                double mc_theta = theta_calculation(mc_px, mc_py, mc_pz)

                double dphi   = Math.abs(exp_phi   - mc_phi)
                double dtheta = Math.abs(exp_theta - mc_theta)

                if (dphi < MC_SCALE*3.0 && dtheta < MC_SCALE*1.0) {
                    matched     = true
                    matching_pid = mcBank.getInt("pid", i)
                    matched_px  = mc_px
                    matched_py  = mc_py
                    matched_pz  = mc_pz
                    match_dphi   = dphi
                    match_dtheta = dtheta
                }
            }
        }

        // Step C (no Step A match): all three stay -9999
        if (!matched) return result

        double mc_match_quality = Math.sqrt(match_dphi*match_dphi + match_dtheta*match_dtheta)
        result[0] = matching_pid
        result[2] = mc_match_quality
        // result[1] (mc_parent_pid) stays -9999 unless Step B succeeds below

        // ── Step B: lookup parent in MC::Lund by PID + momentum ──────────────
        // MC::Particle is a subset of MC::Lund type==1 rows. Find the unique Lund
        // row that matches by PID and momentum (within 1e-4 GeV per component),
        // then read its parent index to obtain mc_parent_pid.
        if (lundBank) {
            for (int i = 0; i < lundBank.rows(); i++) {
                // Only consider final-state (type==1) Lund particles
                int lund_type = (int)(lundBank.getByte("type", i) & 0xFF)
                if (lund_type != 1) continue

                // PID must match the MC::Particle anchor
                if (lundBank.getInt("pid", i) != matching_pid) continue

                // Momentum components must match within tolerance
                double lund_px = lundBank.getFloat("px", i)
                double lund_py = lundBank.getFloat("py", i)
                double lund_pz = lundBank.getFloat("pz", i)
                if (Math.abs(lund_px - matched_px) > 1e-4) continue
                if (Math.abs(lund_py - matched_py) > 1e-4) continue
                if (Math.abs(lund_pz - matched_pz) > 1e-4) continue

                // This is the Lund row for our track — read its parent
                int parent_idx = (int)(lundBank.getByte("parent", i) & 0xFF) - 1
                if (parent_idx >= 0 && parent_idx < lundBank.rows()) {
                    result[1] = lundBank.getInt("pid", parent_idx)
                }
                // else: primary particle (no parent) → mc_parent_pid stays -9999
                break
            }
        }

        return result
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

    // ── Beam energy by run — mirrors BeamEnergy.java without importing analyzers.* ──
    // Uses the CLI-supplied fallback for runnum==11 (MC) and unknown periods.
    static double beamEnergyForRun(int runnum, double fallback) {
        if (runnum == 11) return fallback          // MC: always use CLI value
        if (runnum >= 5032 && runnum <= 5666) return 10.6041   // RGA Fa18
        if (runnum >= 6616 && runnum <= 6783) return 10.1998   // RGA Sp19
        if (runnum >= 6120 && runnum <= 6399) return 10.5986   // RGB Sp19
        if (runnum >= 11093 && runnum <= 11283) return 10.5473 // RGC Su22 (approx)
        return fallback  // unknown period: use CLI default
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ── Main entry point ───────────────────────────────────────════════════════
    // ═══════════════════════════════════════════════════════════════════════════
    public static void main(String[] args) {

        long startTime = System.currentTimeMillis()
        final int N_COLUMNS = 53
        println("=" * 72)
        println("processing_mc_pid_training.groovy  —  ML PID training ntuple")
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

        String output_file = args.length < 2 ? "pid_training_out.txt" : args[1]
        File file = new File(output_file)
        file.delete()

        int n_files = (args.length < 3 ||
                       Integer.parseInt(args[2]) == 0 ||
                       Integer.parseInt(args[2]) > hipo_list.size())
                      ? hipo_list.size() : Integer.parseInt(args[2])

        double beam_energy = args.length < 4 ? 10.6041 : Double.parseDouble(args[3])
        println("Beam energy: ${beam_energy} GeV")

        Integer userProvidedRun = (args.length >= 5) ? Integer.parseInt(args[4]) : null
        if (userProvidedRun == 11) println("MC mode (runnum forced to 11): QA always passes.")

        // ── Physics setup ─────────────────────────────────────────────────────
        // research_fitter used to obtain PhysicsEvent for the Inclusive analyzer
        analysis_fitter research_fitter = new analysis_fitter(10.6041)

        // ── QA database setup (copy pattern from processing_two_particles.groovy) ─
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
        [   // runs with Misc that should be allowed (empty target, RICH-off, etc.)
            6736, 6737, 6738, 6739, 6740, 6741, 6742, 6743, 6744, 6746, 6747,
            6748, 6749, 6750, 6751, 6753, 6754, 6755, 6756, 6757,
            16194, 16089, 16185, 16308, 16184, 16307, 16309,
            16872, 16975,
            17763, 17764, 17765, 17766, 17767, 17768,
            17179, 17180, 17181, 17182, 17183, 17188, 17189, 17252
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
                // Skip Hall-C bleedthrough run range
                if (runnum > 16600 && runnum < 16700) continue
                // Hard upper bound matching processing_two_particles.groovy
                if (runnum > 17768) continue

                // ── QA: runnum==11 (MC) always passes; otherwise consult QADB ──
                boolean passQA = (runnum == 11 || runnum < 5020 || qa.pass(runnum, evnum))
                if (!passQA) continue

                // ── Load all banks for this event ──────────────────────────────
                Map banks = loadBanks(event)
                banks["RUN::config"] = config_bank_raw  // already loaded above

                def rec_bank  = banks["REC::Particle"]
                def run_bank  = banks["RUN::config"]
                if (!rec_bank || !run_bank) continue

                // ── Event-level electron filter ────────────────────────────────
                // Require pid==11 at row 0 and full electron cuts (FD + SF + fiducial)
                if (!passElectronCuts(banks)) continue

                // ── Electron vz (row 0) — needed for passHadronCuts vertex cut ──
                double vz_e = rec_bank.getFloat("vz", 0)

                // ── DIS kinematics via Inclusive analyzer ─────────────────────
                // Mirrors the convention of processing_inclusive.groovy exactly:
                //   BeamEnergy Eb = new BeamEnergy(research_Event, runnum, false)
                //   double energy = (runnum == 11) ? beam_energy : Eb.Eb()
                //   Inclusive variables = new Inclusive(event, research_Event, energy)
                PhysicsEvent research_Event = research_fitter.getPhysicsEvent(event)
                BeamEnergy Eb = new BeamEnergy(research_Event, runnum, false)
                double energy = (runnum == 11) ? beam_energy : Eb.Eb()
                Inclusive inc = new Inclusive(event, research_Event, energy)
                double Q2 = inc.Q2()
                double W  = inc.W()
                double x  = inc.x()
                double y  = inc.y()
                double nu = inc.nu()

                // ── Hadron loop (skip row 0 which is the electron) ─────────────
                for (int row = 1; row < rec_bank.rows(); row++) {

                    int pid = rec_bank.getInt("pid", row)
                    if (!HADRON_PIDS.contains(pid)) continue

                    // ── Cut 1: FD-only + DC-fiducial ──────────────────────────
                    if (!passHadronCuts(row, pid, banks)) continue

                    // ── Per-track kinematics ───────────────────────────────────
                    float h_px = rec_bank.getFloat("px", row)
                    float h_py = rec_bank.getFloat("py", row)
                    float h_pz = rec_bank.getFloat("pz", row)
                    float h_vz = rec_bank.getFloat("vz", row)
                    int h_status = rec_bank.getInt("status", row)

                    // NOTE: No momentum or energy-loss corrections applied to MC.
                    // Corrections are derived for data and would over-correct MC. A separate
                    // processing_data_pid_training.groovy will apply them when run on data.

                    int h_sector = extractSector(row, banks["REC::Track"])
                    double h_p     = Math.sqrt(h_px*h_px + h_py*h_py + h_pz*h_pz)
                    double h_theta = thetaDeg(h_px, h_py, h_pz)
                    double h_phi   = phiDeg(h_px, h_py)

                    // ── REC::Particle scalar features ──────────────────────────
                    double beta     = rec_bank.getFloat("beta",    row)
                    double chi2pid  = rec_bank.getFloat("chi2pid", row)

                    // ── FTOF detector responses ────────────────────────────────
                    double[] ftof = extractFTOF(row, banks["REC::Scintillator"])
                    // [energy_1A, energy_1B, time_1A, time_1B, path_1A, path_1B,
                    //  energy_2,  time_2,    path_2]

                    // ── ECAL / PCAL calorimeter responses ──────────────────────
                    double[] ecal = extractECAL(row, banks["REC::Calorimeter"])
                    // [pcal_e, pcal_t, pcal_path, ecin_e, ecin_t, ecin_path, ecout_e, ecout_t, ecout_path]

                    // ── HTCC nphe ──────────────────────────────────────────────
                    double nphe_htcc = extractHTCC(row, banks["REC::Cherenkov"])

                    // ── RICH variables ─────────────────────────────────────────
                    double[] rich = extractRICH(row, banks["RICH::Particle"])

                    // ── MC truth matching ──────────────────────────────────────
                    double[] mc = extractMCTruth(h_px, h_py, h_pz, banks["MC::Lund"], banks["MC::Particle"])

                    // ── Assemble output row (53 columns; see header + final println) ──
                    int helicity = event.hasBank("REC::Event") ?
                        ((HipoDataBank) event.getBank("REC::Event")).getByte("helicity", 0) : (int)MISSING

                    StringBuilder row_sb = new StringBuilder()
                    // Event-level
                    row_sb.append(runnum).append(' ').append(evnum).append(' ').append(helicity).append(' ')
                    row_sb.append(Q2).append(' ').append(W).append(' ')
                    row_sb.append(x).append(' ').append(y).append(' ').append(nu).append(' ')
                    // Per-track kinematics
                    row_sb.append(pid).append(' ').append(h_p).append(' ').append(h_theta).append(' ')
                    row_sb.append(h_phi).append(' ').append(h_vz).append(' ').append(h_sector).append(' ')
                    row_sb.append(h_status).append(' ')
                    // ML features — 13 features (beta + FTOF 1A/1B + ECAL inner/outer) + chi2pid + nphe_htcc
                    row_sb.append(beta).append(' ').append(chi2pid).append(' ')
                    row_sb.append(ftof[0]).append(' ').append(ftof[1]).append(' ')  // energy 1A, 1B
                    row_sb.append(ftof[2]).append(' ').append(ftof[3]).append(' ')  // time 1A, 1B
                    row_sb.append(ftof[4]).append(' ').append(ftof[5]).append(' ')  // path 1A, 1B
                    row_sb.append(ecal[3]).append(' ').append(ecal[6]).append(' ')  // ecin_e, ecout_e
                    row_sb.append(ecal[4]).append(' ').append(ecal[7]).append(' ')  // ecin_t, ecout_t
                    row_sb.append(ecal[5]).append(' ').append(ecal[8]).append(' ')  // ecin_path, ecout_path
                    row_sb.append(nphe_htcc).append(' ')
                    // PCAL + FTOF layer 2
                    row_sb.append(ecal[0]).append(' ').append(ecal[1]).append(' ').append(ecal[2]).append(' ')
                    row_sb.append(ftof[6]).append(' ').append(ftof[7]).append(' ').append(ftof[8]).append(' ')
                    // RICH (14 vars)
                    rich.each { row_sb.append(it).append(' ') }
                    // MC truth
                    // mc[0] and mc[1] are MC PIDs (integer-valued in the source); cast to int
                    // to ensure the C++ converter (which reads these as /I) parses them.
                    // mc[2] is mc_match_quality, a true double — leave as-is.
                    row_sb.append((int)mc[0]).append(' ').append((int)mc[1]).append(' ').append(mc[2])
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
        println(" ML FEATURES (13 features: beta + FTOF 1A/1B + ECAL inner/outer; + chi2pid + nphe_htcc):")
        println("  16:beta  17:chi2pid")
        println("  18:ftof_energy_1A  19:ftof_energy_1B  20:ftof_time_1A  21:ftof_time_1B")
        println("  22:ftof_path_1A  23:ftof_path_1B")
        println("  24:ecin_energy  25:ecout_energy  26:ecin_time  27:ecout_time")
        println("  28:ecin_path  29:ecout_path")
        println("  30:nphe_htcc")
        println(" PCAL + FTOF LAYER 2 (included for completeness; evaluate for training):")
        println("  31:pcal_energy  32:pcal_time  33:pcal_path")
        println("  34:ftof_energy_2  35:ftof_time_2  36:ftof_path_2")
        println(" RICH CROSS-CHECK (NOT training features):")
        println("  37:rich_emilay  38:rich_emico  39:rich_emqua  40:rich_best_PID")
        println("  41:rich_RQ  42:rich_ReQ")
        println("  43:rich_el_logl  44:rich_pi_logl  45:rich_k_logl  46:rich_pr_logl")
        println("  47:rich_best_ch  48:rich_best_c2  49:rich_best_RL  50:rich_best_ntot")
        println(" MC TRUTH (geometric match |Δφ|<9°, |Δθ|<3°):")
        println("  51:mc_matching_pid  52:mc_parent_pid  53:mc_match_quality")
        println("=" * 72)
        println("Output file: ${output_file}")
        println("Events processed: ${num_events}")
        println("Hadron rows written: ${rows_written}")
        long elapsed = System.currentTimeMillis() - startTime
        println("Elapsed time: ${elapsed} ms")

    } // end main

} // end class PIDTrainingScript
