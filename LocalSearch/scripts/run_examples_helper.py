import os, platform, subprocess, sys

def run_examples(computeCapability):
    #configuration
    if (platform.system() == "Windows" or platform.system().startswith("CYGWIN")):
        print("Using Windows setup ...");
        exe_path = "src/Release/"
        exe_suffix = ".exe"
    elif (platform.system() == "Linux"):
        print("Using Linux setup ...");
        exe_path = "src/bin/"
        exe_suffix = ""
    else:
        print("Unknown platform '{0}'".format(platform.system()))
        exit();

    if (computeCapability == 1): 
        cc_suffix = "__cc10";
    elif (computeCapability == 2):
        cc_suffix = "__cc20";
    else:
        print("Compute capability of {0} not supported!".format(computeCapability))
        exit();

    for p in [1,2,3]:
        print("\n\n========================== Running problem setup {0} ===========================================".format(p))
        parameter = str(p);

        exe = exe_path + "ls_on_cpu_sequential" + exe_suffix;
        print("\n+++ Sequential CPU Version with double loop - ({0}) +++".format(exe));
        exe = [exe] + [parameter]
        sys.stdout.flush()
        subprocess.call(exe)

        exe = exe_path + "ls_on_cpu_openmp_1" + exe_suffix;
        print("\n+++ CPU Version with lexicographical mapping of linear index to cities - ({0}) +++".format(exe));
        exe = [exe] + [parameter]
        if (p == 3):
            print("!!! This version is expected to fail with a segmentation fault !!!");
        sys.stdout.flush()
        subprocess.call(exe)

        exe = exe_path + "ls_on_cpu_openmp_2" + exe_suffix;
        print("\n+++ CPU Version with cut-triangle mapping of linear index to cities - ({0}) +++".format(exe));
        exe = [exe] + [parameter]
        sys.stdout.flush()
        subprocess.call(exe)

        for version in [1,2,3,4,5,6,8]:
            exe = exe_path + "ls_on_gpu_" + str(version) + cc_suffix + exe_suffix
            print("\n--- GPU Version {0} - ({1}) ---".format(version,exe))
            exe = [exe] + [parameter]
            if (p == 3 and version < 8):
                print("!!! This version is expected to fail, resulting in 0 iterations !!!");
            sys.stdout.flush()
            subprocess.call(exe)

    for p in [1,2,3]:
        print("\n\n=================== Running filtering Euclidean Distance Local Search - setup {0} ======================".format(p))
        parameter = str(p);

        exe = exe_path + "ls_on_gpu_8" + cc_suffix + exe_suffix
        print("\n+++ GPU Version with no filtering implemented - ({0}) +++".format(exe));
        exe = [exe] + [str(p+3)]
        sys.stdout.flush()
        subprocess.call(exe)

        for r in [0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995]:
            print("\n-------------- Filter {0} % --------------".format(100.0*r))
            for version in [1,3]:
                exe = exe_path + "ls_on_gpu_filter_" + str(version) + cc_suffix + exe_suffix
                print("\n--- GPU Filter Version {0} - ({1}) ---".format(version,exe))
                exe = [exe] + [parameter] + [str(r)]
                sys.stdout.flush()
                subprocess.call(exe)

    for p in [1,2,3]:
        print("\n\n=================== Running filtering Compute Intensive Distance Local Search - setup {0} ======================".format(p))
        parameter = str(p);

        exe = exe_path + "ls_on_gpu_9" + cc_suffix + exe_suffix
        print("\n+++ GPU Version with no filtering implemented - ({0}) +++".format(exe));
        exe = [exe] + [parameter]
        sys.stdout.flush()
        subprocess.call(exe)

        for r in [0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995]:
            print("\n-------------- Filter {0} % --------------".format(100.0*r))
            for version in [2,4]:
                exe = exe_path + "ls_on_gpu_filter_" + str(version) + cc_suffix + exe_suffix
                print("\n--- GPU Filter Version {0} - ({1}) ---".format(version,exe))
                exe = [exe] + [parameter] + [str(r)]
                sys.stdout.flush()
                subprocess.call(exe)


