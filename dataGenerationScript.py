import numpy as np
import sys
import subprocess

args = sys.argv
batch_count = int(args[1])
# params_file = args[2]

# i = 0
# with open(params_file) as f:
#     lines = f.readlines()
#     for x in range(batch_count):
#         for l in lines:
#             i += 1
#             print(str(i) + "/" + str(batch_count * len(lines)))
#             subprocess.run("export batchId=" + str(x) + " ; " + l, shell=True)

# variantsStructure = ["a","b","c","d"]
# variantsCluCount = ["3", "4", "5", "6", "7", "8"]
dataBasePath = "q2"
ssmBasePath = "q2-ssm"
tsneBasePath = "q2-tsne"

def generateData(batchCount, datasetSize, dimSize, spCount, spSizes, cluSeps, spShares, cluSizeVars, missRate, cluDensity, cluExclude, structId):
    cluCount = 0
    for xx in spSizes:
        cluCount += xx
    for batch in range(batchCount):
        
        coreFilename = str(cluCount) + "-" + structId + "-" + str(batch)
        baseFile = dataBasePath + "/" + coreFilename + ".h5"
        logFile = dataBasePath + "/" + coreFilename + ".log"
        projFile = dataBasePath + "/" + coreFilename + ".png"
        ssmOutputFile = ssmBasePath + "/" + coreFilename + "-ssm.h5"
        tsneOutputFile = tsneBasePath + "/" + coreFilename + "-tsne.h5"
        ssmProjOutputPrefix = ssmBasePath + "/" + coreFilename + "-ssm"
        tsneProjOutputPrefix = tsneBasePath + "/" + coreFilename + "-tsne"
        gtProjOutputSuffix = "-gt.png"
        distProjOutputSuffix = "-dist.png"
        spSizes = [str(x) for x in spSizes]
        cluSeps = [str(x) for x in cluSeps]
        spShares = [str(x) for x in spShares]
        cluSizeVars = [str(x) for x in cluSizeVars]
        cluDensity = [str(x) for x in cluDensity]
        cluExclude = [str(x) for x in cluExclude]
        commandString = "./nnSsmDemo --generate --size " + str(datasetSize) + " --dims " + str(dimSize) + " --sp " + str(spCount) + " --spSize " + ",".join(spSizes) + " --cluSep " + ",".join(cluSeps) + " --spShare " + ",".join(spShares) + " --cluSizeVar " + ",".join(cluSizeVars)
        if missRate > 0:
            commandString += " --missRate " + str(missRate)
        if len(cluDensity) > 0:
            commandString += " --cluDensity " + ",".join(cluDensity)
        if len(cluExclude) > 0:
            commandString += " --cluExclude " + ",".join(cluExclude)

        commandString += " --output " + baseFile + " 2> " + logFile
        print(commandString)
        subprocess.run(commandString, shell=True)
        #subprocess.run("./nnSsmDemo --proj " + baseFile, shell=True)
        #subprocess.run("convert capture.ppm capture.png", shell=True)
        #subprocess.run("mv capture.png " + projFile, shell=True)
        #subprocess.run("./nnSsmDemo --ssm-cpp 1 " + baseFile + " --output " + ssmOutputFile, shell=True)
        subprocess.run("./nnSsmDemo --tsne 1 " + baseFile + " --output " + tsneOutputFile, shell=True)
        #subprocess.run("./nnSsmDemo --map " + ssmOutputFile + " --gt", shell=True)
        #subprocess.run("convert capture.ppm capture.png", shell=True)
        #subprocess.run("mv capture.png " + ssmProjOutputPrefix + gtProjOutputSuffix, shell=True)
        #subprocess.run("./nnSsmDemo --map " + ssmOutputFile + " --revviridis", shell=True)
        #subprocess.run("convert capture.ppm capture.png", shell=True)
        #subprocess.run("mv capture.png " + ssmProjOutputPrefix + distProjOutputSuffix, shell=True)
        subprocess.run("./nnSsmDemo --map " + tsneOutputFile + " --gt", shell=True)
        subprocess.run("convert capture.ppm capture.png", shell=True)
        subprocess.run("mv capture.png " + tsneProjOutputPrefix + gtProjOutputSuffix, shell=True)
        subprocess.run("./nnSsmDemo --map " + tsneOutputFile + " --revviridis", shell=True)
        subprocess.run("convert capture.ppm capture.png", shell=True)
        subprocess.run("mv capture.png " + tsneProjOutputPrefix + distProjOutputSuffix, shell=True)





batchCount = batch_count
missRates = [0]
# struct a
for mr in missRates:
    # for cluCount in range(3, 9):
    #     generateData(batchCount, 128, 3, 1, [cluCount], [1.4], [1], [np.random.random() * 0.4], mr, [], [], "a" + str(mr))

    # # struct b
    # generateData(batchCount, 128, 3, 2, [3, 2], [1.4, 0.2], [3, 2], [0.3, 0.4], mr, ["3:0.8"], ["1:1:0"], "b" + str(mr))
    # generateData(batchCount, 128, 3, 3, [3, 2, 2], [1.4, 0.2, 0.2], [3, 2, 2], [0.3, 0.4, 0.4], mr, ["3:0.8", "5:0.8"], ["1:1:0", "2:1:0"], "b" + str(mr))

    # # struct c
    # generateData(batchCount, 128, 3, 2, [3, 2], [1.3, 0.6], [3, 2], [0.3, 0.4], mr, [], [], "c" + str(mr))
    # generateData(batchCount, 128, 3, 3, [3, 2, 2], [1.3, 0.6, 0.6], [3, 2, 2], [0.3, 0.4, 0.4], mr, [], [], "c" + str(mr))

    # #struct d
    # generateData(batchCount, 128, 3, 3, [1, 2, 2], [1.3, 1.3, 1.3], [1, 2, 2], [0.3, 0.3, 0.3], mr, [], [], "d" + str(mr))
    # generateData(batchCount, 128, 3, 4, [1, 2, 2, 2], [1.3, 1.3, 1.3, 1.3], [1, 2, 2, 2], [0.3, 0.3, 0.3, 0.3], mr, [], [], "d" + str(mr))
    generateData(batchCount, 64, 7, 1, [7], [1.4], [1], [np.random.random() * 0.4], mr, [], [], "a" + str(mr))


# for b in range(batch_count):
#     for x in variantsStructure:
#         for y in variantsCluCount:
#             coreFilename = y + "-" +    x + "-" + str(b)
#             baseFile = dataBasePath + "/" + coreFilename + ".h5"
#             ssmOutputFile = ssmBasePath + "/" + coreFilename + "-ssm.h5"
#             tsneOutputFile = tsneBasePath + "/" + coreFilename + "-tsne.h5"
#             subprocess.run("./nnSsmDemo --ssm-cpp 1 " + baseFile + " --output " + ssmOutputFile, shell=True)
#             subprocess.run("./nnSsmDemo --tsne 1 " + baseFile + " --output " + tsneOutputFile, shell=True)
#             subprocess.run("./nnSsmDemo --map " + ssmOutputFile + " --gt", shell=True)
#             subprocess.run("convert capture.ppm capture.png", shell=True)
#             subprocess.run("mv capture.png " + ssmBasePath + "/" + coreFilename + "-ssm-gt.png", shell=True)
#             subprocess.run("./nnSsmDemo --map " + ssmOutputFile + " --revviridis", shell=True)
#             subprocess.run("convert capture.ppm capture.png", shell=True)
#             subprocess.run("mv capture.png " + ssmBasePath + "/" + coreFilename + "-ssm-dist.png", shell=True)
#             subprocess.run("./nnSsmDemo --map " + tsneOutputFile + " --gt", shell=True)
#             subprocess.run("convert capture.ppm capture.png", shell=True)
#             subprocess.run("mv capture.png " + tsneBasePath + "/" + coreFilename + "-tsne-gt.png", shell=True)
#             subprocess.run("./nnSsmDemo --map " + tsneOutputFile + " --revviridis", shell=True)
#             subprocess.run("convert capture.ppm capture.png", shell=True)
#             subprocess.run("mv capture.png " + tsneBasePath + "/" + coreFilename + "-tsne-dist.png", shell=True)
