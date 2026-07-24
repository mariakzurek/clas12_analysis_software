import org.jlab.io.hipo.HipoDataSource
import org.jlab.io.hipo.HipoDataBank
import extended_kinematic_fitters.generic_tests
import extended_kinematic_fitters.fiducial_cuts
import extended_kinematic_fitters.pid_cuts

def GENERIC_TESTS = new generic_tests()
def FIDUCIAL_CUTS = new fiducial_cuts()
def PID_CUTS      = new pid_cuts()

def connorElectronVertexCut = { HipoDataBank rec, HipoDataBank run ->
    if (!rec || !run) return false
    if (rec.rows() == 0) return false
    double vz_e  = rec.getFloat("vz", 0)
    float  torus = run.getFloat("torus", 0)
    boolean inbending = (torus <= 0)
    double vz_lo = inbending ? -8.0 : -11.0
    double vz_hi = inbending ?  2.0 :   1.0
    return (vz_e > vz_lo && vz_e < vz_hi)
}

def connorHadronVertexCut = { int row, HipoDataBank rec ->
    if (!rec) return false
    if (rec.rows() == 0 || row >= rec.rows()) return false
    double vz_e = rec.getFloat("vz", 0)
    double vz_h = rec.getFloat("vz", row)
    return (Math.abs(vz_e - vz_h) < 20.0)
}

def passElectronCuts = { Map banks ->
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

def file = "/cache/clas12/rg-a/production/montecarlo/clasdis_pass2/fa18_inb/Q2_1.5GeV/nb-clasdis-Q2_1.5-10023_0.hipo"
final long MAX_EVENTS = 20000

long eventsScanned        = 0
long eventsPassElectron   = 0

long n_pid0_total         = 0
long n_pid0_chargepos     = 0
long n_pid0_electrongate  = 0
long n_pid0_fd            = 0
long n_pid0_vertex        = 0
long n_pid0_dcfid         = 0

def reader = new HipoDataSource()
reader.open(file)

while (reader.hasEvent() && eventsScanned < MAX_EVENTS) {
    def event = reader.getNextEvent()
    eventsScanned++

    def rec  = event.hasBank("REC::Particle")    ? (HipoDataBank) event.getBank("REC::Particle")    : null
    def cal  = event.hasBank("REC::Calorimeter") ? (HipoDataBank) event.getBank("REC::Calorimeter") : null
    def traj = event.hasBank("REC::Traj")        ? (HipoDataBank) event.getBank("REC::Traj")        : null
    def run  = event.hasBank("RUN::config")      ? (HipoDataBank) event.getBank("RUN::config")      : null

    if (!rec || !run) continue

    Map banks = [
        "REC::Particle":    rec,
        "REC::Calorimeter": cal,
        "REC::Traj":        traj,
        "RUN::config":      run
    ]

    boolean electronPass = passElectronCuts(banks)
    if (electronPass) eventsPassElectron++

    for (int row = 1; row < rec.rows(); row++) {
        if (rec.getInt("pid", row) != 0) continue
        n_pid0_total++

        if (rec.getByte("charge", row) <= 0) continue
        n_pid0_chargepos++

        if (!electronPass) continue
        n_pid0_electrongate++

        if (!GENERIC_TESTS.forward_detector_cut(row, rec)) continue
        n_pid0_fd++

        if (!connorHadronVertexCut(row, rec)) continue
        n_pid0_vertex++

        if (!traj || !FIDUCIAL_CUTS.dc_fiducial_cut(row, rec, traj, run)) continue
        n_pid0_dcfid++
    }
}

reader.close()

println "=" * 60
println "File: ${file}"
println "Events scanned              : ${eventsScanned}"
println "Events passing electron gate: ${eventsPassElectron}"
println ""
println "pid==0 cutflow (cumulative rows):"
println "  total pid==0 (row>0)      : ${n_pid0_total}"
println "  + charge > 0              : ${n_pid0_chargepos}"
println "  + electron gate           : ${n_pid0_electrongate}"
println "  + forward_detector_cut    : ${n_pid0_fd}"
println "  + hadron vertex cut       : ${n_pid0_vertex}"
println "  + dc_fiducial_cut         : ${n_pid0_dcfid}"
println "=" * 60
