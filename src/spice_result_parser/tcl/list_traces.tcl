# list_traces.tcl: list the signals of one waveform file (WaveView batch mode).
#
# Part of spice_result_parser (waveview.py runs it). Run a copy of it by hand in
# a working directory (WaveView writes ace_tcl.log beside the script):
#   Linux:   SRP_IN=run.tr0 SRP_OUT=out.txt wv -ace_no_gui list_traces.tcl
#   Windows: set SRP_IN=run.tr0&& set SRP_OUT=out.txt&& wv.exe -ace_no_gui list_traces.tcl
#
# Input:  SRP_IN  the waveform file (HSPICE .tr#/.sw#/.ac#, Nutmeg .raw, ...).
# Output: SRP_OUT a text file, one record per line, fields separated by a tab:
#   srp_listing 1
#   title   <title line, trimmed>
#   xname   <x-axis name of the first signal>
#   param   <name>          one per sweep parameter, in order
#   sweep   <value> ...     one per trace: the value of each parameter, in the
#                           order of the param records (%g, 6 significant
#                           digits); none when not swept
#   signal  <name> <real|complex> <npoints>
#                           npoints: WaveView's point count of the signal; in a
#                           swept file it holds one more point per sweep
#   srp_done
# A file that WaveView cannot open gives "error<TAB>cannot open <path>".
# sx_open_sim_file_read splits its argument as a Tcl list, so the path goes in
# as one list element: [list] for most paths, in quotes when it has a brace
# ([list] escapes a lone brace with a backslash, which WaveView changes to "/").
# WaveView cannot open a path with non-ASCII characters, or a Windows path of
# 260 characters or more: the script then uses the 8.3 short name of the path.
# The last line is always srp_done. When it is missing, the script failed: the
# console (stdout) has the WaveView error. The exit code does not show success:
# it is 0 after an error that the script catches.
#
# Tcl 8.4: no dict, lassign, {*} or "in".

set out [open $env(SRP_OUT) w]
fconfigure $out -encoding utf-8

proc emit {args} {
    global out
    puts $out [join $args "\t"]
}

proc one_line {text} {
    return [string trim [string map [list "\t" " " "\r" " " "\n" " "] $text]]
}

# A name as valid UTF-8: WaveView gives the bytes of the file, and puts can lose
# the end of a line that holds a Latin-1 byte.
proc clean {name} {
    return [encoding convertfrom utf-8 [encoding convertto utf-8 $name]]
}

emit srp_listing 1
set path $env(SRP_IN)
if {![string is ascii $path] || [string length $path] >= 260} {
    catch {set path [file attributes $path -shortname]}
}
set arg [list $path]
if {[regexp {[{}]} $path] && [string first "\"" $path] < 0} {
    set arg "\"$path\""
}
set files {}
catch {set files [sx_open_sim_file_read $arg]}
set f [lindex $files 0]
set why ""
if {![string is ascii $path]} {
    set why ": WaveView cannot open a path with non-ASCII characters"
} elseif {[string length $path] >= 260} {
    set why ": WaveView cannot open a path of 260 characters or more"
}
if {$f == ""} {
    emit error "cannot open $env(SRP_IN)$why"
} else {
    emit title [one_line [clean [sx_wdf_info $f -title]]]
    sx_current_sim_file $f
    set sigs [sx_signal *]
    if {[llength $sigs] > 0} {
        emit xname [clean [sx_signal_attribute [lindex $sigs 0] -xname]]
    }
    foreach p [sx_get_sweep_parameters $f] {
        emit param [clean $p]
    }
    foreach c [sx_get_sweep_conditions $f] {
        emit sweep [join $c "\t"]
    }
    foreach s $sigs {
        emit signal [clean [sx_get_name $s]] [sx_signal_attribute $s -datatype] \
            [sx_signal_attribute $s -npoints]
    }
}
emit srp_done
close $out
exit
