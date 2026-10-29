# read_measures.tcl: read the columns of an HSPICE measure file (.mt#, .ms#,
# .ma#) point by point (WaveView batch mode).
#
# Part of spice_result_parser (waveview.py runs it). Run a copy of it by hand in
# a working directory (WaveView writes ace_tcl.log beside the script):
#   Linux:   SRP_IN=run.mt0 SRP_OUT=out.txt wv -ace_no_gui read_measures.tcl
#   Windows: set SRP_IN=run.mt0&& set SRP_OUT=out.txt&& wv.exe -ace_no_gui read_measures.tcl
#
# WaveView opens a measure file as waveforms: x is "index" and each column is a
# signal ("index" too, when the file has an index column; "alter#" is not one).
# The scan commands are used, not sx_export_data: the export repeats the first row.
#
# Input:  SRP_IN      the measure file.
#         SRP_SIGNALS optional: a text file with one exact column name per line.
#                     Not set: every column.
# Output: SRP_OUT     a text file, fields separated by a tab:
#   srp_measures 1
#   title   <title line, trimmed>
#   column  <name>      then one line per row with the value (17 digits)
#   ...                 (in the order of the file; a name can come twice)
#   srp_done
# A failed measure reads as NaN (1.#QNAN on Windows). A file that does not open
# or an unknown column gives "error<TAB><text>" then srp_done. A missing
# srp_done means that the script failed: the console (stdout) has the WaveView
# error. The exit code does not show success: it is 0 after an error that the
# script catches.
#
# Tcl 8.4: no dict, lassign, {*} or "in".

set tcl_precision 17
set out [open $env(SRP_OUT) w]
fconfigure $out -encoding utf-8

proc finish {{error ""}} {
    global out
    if {$error != ""} {
        puts $out "error\t$error"
    }
    puts $out srp_done
    close $out
    exit
}

# Write the y value of each point of signal `sig`, one per line. The loop is in
# a proc so that Tcl compiles it.
proc write_values {sig} {
    global out
    set scan [sx_scan_data_points $sig POINT]
    set more [sx_jump_to_x $scan FIRST]
    while {$more == 1} {
        puts $out [sx_y_value $scan]
        set more [sx_jump_to_x $scan NEXT]
    }
    sx_release $scan
}

puts $out "srp_measures\t1"
# sx_open_sim_file_read splits its argument as a Tcl list: pass one element, in
# quotes when the path has a brace ([list] escapes a lone brace with a backslash,
# which WaveView changes to "/"). It cannot open a path with non-ASCII characters
# or a Windows path of 260 characters or more: try the 8.3 short name.
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
    finish "cannot open $env(SRP_IN)$why"
}
set title [sx_wdf_info $f -title]
puts $out "title\t[string trim [string map [list "\t" " " "\r" " " "\n" " "] $title]]"
sx_current_sim_file $f
# Two columns can have one name (a measure named as a .data parameter): keep
# the signal objects in a list, not only in the array of names.
set all [sx_signal *]
foreach s $all {
    set object([sx_get_name $s]) $s
}

if {[info exists env(SRP_SIGNALS)]} {
    set in [open $env(SRP_SIGNALS) r]
    fconfigure $in -encoding utf-8
    while {[gets $in line] >= 0} {
        set name [string trimright $line "\r"]
        if {$name == ""} {
            continue
        }
        if {![info exists object($name)]} {
            close $in
            finish "unknown signal $name"
        }
        set wanted($name) 1
    }
    close $in
} else {
    foreach name [array names object] {
        set wanted($name) 1
    }
}

foreach s $all {
    set name [sx_get_name $s]
    if {![info exists wanted($name)]} {
        continue
    }
    puts $out "column\t$name"
    write_values $s
}
finish
