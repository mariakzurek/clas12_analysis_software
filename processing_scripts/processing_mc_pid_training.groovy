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
 *  1. Requires a trigger electron at REC::Particle row 0 that passes
 *     analysis_fitter.electron_test() — same cuts as the production SIDIS analysis.
 *  2. Loops over every other REC::Particle row; writes ONE ROW per FD charged
 *     hadron track with pid ∈ {+211, -211, +321, -321} that also passes:
 *       (a) generic_tests.forward_detector_cut()  — |status| ∈ [2000, 4000)
 *       (b) analysis_fitter.pion_test()  (for |pid| == 211)
 *           analysis_fitter.kaon_test()  (for |pid| == 321)
 *           Both enforce vertex + DC-fiducial; chi2pid is NOT a cut.
 *       (c) Energy-loss and momentum corrections applied to pion tracks (π+: stefan +
 *           Capobianco; π-: krishna). NO correction is applied for K± because the
 *           Hayward framework has no kaon corrector; raw momentum is written and this
 *           limitation is noted in the output header. Update this script if a kaon
 *           corrector is added to energy_loss_corrections.java.
 *  3. Reads per-track detector-response variables directly from HIPO banks (no
 *     analyzer Java class is used — avoids the getIndex() index-alignment bug).
 *  4. Geometrically matches each reconstructed hadron to MC::Lund truth particles
 *     using |Δφ| < 9° AND |Δθ| < 3° (same geometric window as processing_mc_three_particles.groovy).
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
import extended_kinematic_fitters.energy_loss_corrections
import extended_kinematic_fitters.momentum_corrections
import analyzers.BeamEnergy
import analyzers.Inclusive

// ─── Main class ───────────────────────────────────────────────────────────────
public class PIDTrainingScript {

    // ── Constants ──────────────────────────────────────────────────────────────
    static final double MISSING = -9999.0
    // MC truth matching scale factor — matches processing_mc_three_particles.groovy convention
    static final double MC_SCALE = 3.0      // phi window = scale*3°, theta window = scale*1°
    // Allowed hadron PIDs
    static final Set<Integer> HADRON_PIDS = [211, -211, 321, -321] as Set

    // ── Bank loading — returns map of name→bank; null if bank absent ──────────
    static Map<String, HipoDataBank> loadBanks(HipoDataEvent event) {
        def banks = [:]
        ["REC::Particle", "REC::Calorimeter", "REC::Scintillator",
         "REC::Cherenkov", "REC::Track", "REC::Traj",
         "REC::Event", "RUN::config", "MC::Lund", "RICH::Particle"].each { name ->
            banks[name] = event.hasBank(name) ? (HipoDataBank) event.getBank(name) : null
        }
        return banks
    }

    // ── Electron filter — requires pid==11 at row 0 + full electron_test() ────
    // electron_test(int idx, double p, rec, cal, traj, run, cc)  [af.java:24-41]
    static boolean passElectronCuts(Map banks, analysis_fitter af) {
        def rec  = banks["REC::Particle"]
        def cal  = banks["REC::Calorimeter"]
        def traj = banks["REC::Traj"]
        def run  = banks["RUN::config"]
        def cc   = banks["REC::Cherenkov"]
        if (!rec || !cal || !traj || !run || !cc) return false
        if (rec.rows() == 0 || rec.getInt("pid", 0) != 11) return false
        float px = rec.getFloat("px", 0)
        float py = rec.getFloat("py", 0)
        float pz = rec.getFloat("pz", 0)
        double p_e = Math.sqrt(px*px + py*py + pz*pz)
        return af.electron_test(0, p_e, rec, cal, traj, run, cc)
    }

    // ── Per-hadron cut filter — FD-only + vertex + DC-fiducial ───────────────
    // pion_test / kaon_test(int idx, int pid, float vz, double vz_e, rec, cal, traj, run)
    // NOTE: trigger_electron_vz is accepted by the signature but unused in current impl.
    static boolean passHadronCuts(int row, int pid, double vz_e, Map banks, analysis_fitter af) {
        def rec  = banks["REC::Particle"]
        def cal  = banks["REC::Calorimeter"]
        def traj = banks["REC::Traj"]
        def run  = banks["RUN::config"]
        if (!rec || !cal || !traj || !run) return false
        // Step 1: FD-only (|status| ∈ [2000, 4000))
        generic_tests gt = new generic_tests()
        if (!gt.forward_detector_cut(row, rec)) return false
        // Step 2: vertex + DC fiducial via the appropriate test
        float vz_h = rec.getFloat("vz", row)
        int absPid = Math.abs(pid)
        if (absPid == 211) return af.pion_test(row, pid, vz_h, vz_e, rec, cal, traj, run)
        if (absPid == 321) return af.kaon_test(row, pid, vz_h, vz_e, rec, cal, traj, run)
        return false
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

    // ── MC truth matching — pattern from processing_mc_three_particles.groovy ─
    // Returns [mc_matching_pid, mc_parent_pid, mc_match_quality(deg)].
    // mc_parent_pid: PDG of parent (MC::Lund "parent" = 1-based index, 0=primary → -9999).
    static double[] extractMCTruth(double h_px, double h_py, double h_pz, HipoDataBank lundBank) {
        double[] result = [MISSING, MISSING, MISSING]
        if (!lundBank) return result

        boolean matching_h = false
        int matching_h_pid = 0
        int mc_h_parent_index = 0
        double match_dphi = MISSING
        double match_dtheta = MISSING

        for (int current_part = 0; current_part < lundBank.rows(); current_part++) {
            int pid = lundBank.getInt("pid", current_part)
            if (matching_h) { continue }
            double mc_px = lundBank.getFloat("px", current_part)
            double mc_py = lundBank.getFloat("py", current_part)
            double mc_pz = lundBank.getFloat("pz", current_part)

            double mc_phi       = phi_calculation(mc_px, mc_py)
            double mc_theta_lab = theta_calculation(mc_px, mc_py, mc_pz)

            double exp_phi   = phi_calculation(h_px, h_py)
            double exp_theta = theta_calculation(h_px, h_py, h_pz)

            matching_h = Math.abs(exp_phi - mc_phi) < MC_SCALE*3.0 &&
                         Math.abs(exp_theta - mc_theta_lab) < MC_SCALE*1.0
            if (matching_h) {
                matching_h_pid     = pid
                mc_h_parent_index  = lundBank.getInt("parent", current_part) - 1
                match_dphi         = Math.abs(exp_phi - mc_phi)
                match_dtheta       = Math.abs(exp_theta - mc_theta_lab)
            }
        }

        if (!matching_h) return result
        result[0] = matching_h_pid
        // parent field in MC::Lund is a 1-based index; 0 means primary (no parent)
        if (mc_h_parent_index >= 0 && mc_h_parent_index < lundBank.rows()) {
            result[1] = lundBank.getInt("pid", mc_h_parent_index)
        }
        // else: primary particle → mc_parent_pid stays -9999
        result[2] = Math.sqrt(match_dphi*match_dphi + match_dtheta*match_dtheta)
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
        analysis_fitter af = new analysis_fitter(beam_energy)
        energy_loss_corrections elc = new energy_loss_corrections()
        momentum_corrections mc_corr = new momentum_corrections()
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
                // Require pid==11 at row 0 and full electron_test (FD + SF + fiducial)
                if (!passElectronCuts(banks, af)) continue

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

                    // ── Cut 1: FD-only + vertex + DC-fiducial ─────────────────
                    if (!passHadronCuts(row, pid, vz_e, banks, af)) continue

                    // ── Per-track kinematics ───────────────────────────────────
                    float h_px = rec_bank.getFloat("px", row)
                    float h_py = rec_bank.getFloat("py", row)
                    float h_pz = rec_bank.getFloat("pz", row)
                    float h_vz = rec_bank.getFloat("vz", row)
                    int h_status = rec_bank.getInt("status", row)

                    // ── Energy-loss + momentum corrections for pions ───────────
                    // LIMITATION: No kaon corrector exists in energy_loss_corrections.java
                    // or momentum_corrections.java. K± are written with raw momentum.
                    // Update here if a kaon corrector is added to the framework.
                    float[] momentum = [h_px, h_py, h_pz] as float[]
                    int runnum_for_corr = runnum
                    int runPeriod = -1
                    if (runnum_for_corr >= 4763 && runnum_for_corr <= 5666) runPeriod = 3  // RGA Fa18
                    else if (runnum_for_corr >= 6616 && runnum_for_corr <= 6783) runPeriod = 2  // RGA Sp19

                    int h_sector = extractSector(row, banks["REC::Track"])
                    boolean inbending  = (run_bank.getFloat("torus", 0) <= 0)
                    boolean outbending = !inbending

                    if (pid == 211 && h_sector >= 1 && h_sector <= 6) {
                        // π+: Stefan energy-loss correction, then Capobianco momentum correction
                        elc.stefan_piplus_energy_loss_corrections(row, momentum, rec_bank, run_bank, banks["REC::Track"])
                        if (inbending) {
                            mc_corr.inbending_momentum_corrections(momentum, h_sector, 1, 0, runPeriod, 0, 0)
                        } else if (outbending) {
                            mc_corr.outbending_momentum_corrections(momentum, h_sector, 1, 0, runPeriod, 0, 0)
                        }
                    } else if (pid == -211) {
                        // π-: Krishna energy-loss correction (no Capobianco π- momentum correction exists)
                        elc.krishna_energy_loss_corrections(row, momentum, rec_bank, run_bank)
                    }
                    // K+, K-: no corrections — raw momentum written (see LIMITATION above)

                    h_px = momentum[0]; h_py = momentum[1]; h_pz = momentum[2]
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
                    double[] mc = extractMCTruth(h_px, h_py, h_pz, banks["MC::Lund"])

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
                    row_sb.append(mc[0]).append(' ').append(mc[1]).append(' ').append(mc[2])
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
