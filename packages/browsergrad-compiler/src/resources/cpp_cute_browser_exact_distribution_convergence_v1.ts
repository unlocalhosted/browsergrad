import {
  deepFreezeJson,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

const VALUE = {
  schema:
    "browsergrad.compiler.cpp-cute.browser-exact-distribution-convergence-observation",
  version: 1,
  authority:
    "package-pinned-local-engineering-exact-payload-convergence-observation-only",
  matrixId:
    "bg.cpp.browser-exact-distribution-convergence.sha256.12665a3d1f38689f9439f0aba2a6b4c87f42d28e021c5804a528b294f8f7fe31",
  sourceRevision: "8d7f27eb9a249d8277def3b401377c42e961b6c7",
  caseCount: 8,
  distribution: {
    assetManifestId:
      "bg.cpp.browser-assets.sha256.9db5c28897a9d9fd512056a767ade5446e0c188e3dc3f12946929f6d59d01c25",
    assetManifestSha256:
      "40085018f1266909688a6aadead0e1d0dbeea60274bf4113d966cddebcbca10c",
    assetSetSha256:
      "2a3a9256bcc1501ee8fc1bddde10b51a5e7eb59048c5f34e5285cba6383510bd",
    buildInputLockId:
      "bg.cpp.browser-build-input-lock.sha256.fa21cfe45dec6b4869662cd613a7a300848657518f375c04f7f2193f3a874ad4",
    buildInputLockResourceSha256:
      "fd0f4f978399c6e52ebdb0489f35ce6b0a88e289dce8cfdfa112e52d6217cf3c",
    buildSubjectId:
      "bg.cpp.browser-build-subject.sha256.ed6344d3d2ffef5745f92e0ee53d4839e6c6ed7e193a56ca700c61853eb2da98",
    buildSubjectSha256:
      "ed6344d3d2ffef5745f92e0ee53d4839e6c6ed7e193a56ca700c61853eb2da98",
    exactOutputByteLength: "103637695",
    exactOutputCount: 25,
    exactRootVerificationId:
      "bg.cpp.distribution-output-file-verification.sha256.db551532db60b4bf06353ded2140a07731185b9dfbb5b0dd0cca1da67c002b89",
    profileByteLength: "7148",
    profileHash:
      "4f4b7416ec509ea97b612cc5b6c6c01596624ef63b8badc4f2a21ffd6b2e1003",
    profileSha256:
      "16d47a72abe1851ce51810898cfd7d4223eae8e114c1f7ea300858486b30c6a8",
    reproducibilityId:
      "bg.cpp.browser-full-distribution-reproducibility.sha256.64cc7401523b6026aba9430e2f081d708bc62cfcbe6fc343bf58ec0798aeec7b",
    resourceSha256:
      "285a73cc67ec4196104389f127f6728e531aa41c44fba8a79d61e89533f2b76e",
    workerBundleSha256:
      "9c9591e725fca512d10a366bdec38b0067366f3d8ebdef50c29a5ebb0134def5",
  },
  producer: {
    builderId:
      "https://builders.browsergrad.dev/local-engineering-reproducibility",
    keyId:
      "sha256:7071c0ccf9d719c9c089d5a804639cf851bea3b94515fe26cd5d8aa7b1820609",
    policyId:
      "bg.cpp.browser-producer-trust-policy.sha256.dffbe7a05187b43db4b1415384947acd21509bcc49563ea04804b49c5474caa6",
    policySha256:
      "991481989277975c1e673aee58f18b445f941842fa55944f95ae0cf66abcda07",
    producerEvidenceId:
      "bg.cpp.browser-build-producer.sha256.a299565c89b95e92238f4724a3c5c0083ea88105753d997948adb6a9c334a1e8",
    signatureEvidenceSha256:
      "a1be096caea09eea8cb75153d2747b080f0b2daddfa91912d0b1d8e1ff334ec8",
    statementSha256:
      "818b8efae4e9447488ef10c50437699cbf27585a7d722257370192b8aeff86ff",
    trustStoreSha256:
      "6d73beacb22e53675c346e4d9833ac57c38e4861e4cca131fd137fa0ea420804",
  },
  webgpu: {
    actualExecutionObservedForEveryCase: true,
    deviceProfileCount: 1,
    deviceProfileHashes: [
      "823ff00eae2f4ee770ad55563920a43566dd904664915b855f7728c99c6e625f",
    ],
    required: true,
  },
  cases: [
    {
      caseId: "rank2",
      evidenceId:
        "bg.cpp.browser-exact-distribution-case-convergence.sha256.74af30a5c672059803e4851608fdea143b9879a6d46f12c4091ef347f4d9477f",
      sourceSha256:
        "4134804a9892ed1f0a2778fae305e957b5a981afccf2a096f1585f3b1d4e6f06",
      dtype: "f32",
      coordinateRank: 2,
      candidateId:
        "bg.cpp.browser-worker-view-copy-candidate.sha256.0d43ba900061541314472607aa26f0123fb2c0bac95598e353c9908912798002",
      artifactId:
        "bg.artifact.cpp-cute-frontend.sha256.b65cf56763747610e1fc2e726b3cdc03af6f04f0b4825bd9336eff355dba6ad5",
      authorizationId:
        "bg.cpp.browser-view-copy-authorization.sha256.7933822a839841e3db36850a8222a666eaeacbb35f967dd684b73f6a7df591c1",
      executionEvidenceId:
        "bg.cpp.browser-worker-execution.sha256.a182e559b7cc4bc4667b072d4604b87a63999366840113073826ab1b41ff0a6d",
      layoutSemanticHash:
        "5ade6e063773ba40a1046423e76776cf963544a26c7f17b301565d54a86ecdfe",
      kernelSemanticHash:
        "64dc9d67e4f0de9c1f7b68fa369957c9521d8bbb9aa9725ac82f0dfaa573f409",
      cpuDestinationHash:
        "db0e59ae7b62597986196566d4097087f097224e7ba9ba7bce497839dab6764c",
      webGpuDestinationHash:
        "db0e59ae7b62597986196566d4097087f097224e7ba9ba7bce497839dab6764c",
      deviceProfileHash:
        "823ff00eae2f4ee770ad55563920a43566dd904664915b855f7728c99c6e625f",
    },
    {
      caseId: "rank3",
      evidenceId:
        "bg.cpp.browser-exact-distribution-case-convergence.sha256.c6e1cbff82e119a1b8f5b8d3e987d2411498eb4be4b83f1b9f2012fe561c5d25",
      sourceSha256:
        "6a7beae44e88d7fe8749cb5b485dc7d51d30ed285d33314895be461d428550dd",
      dtype: "f32",
      coordinateRank: 3,
      candidateId:
        "bg.cpp.browser-worker-view-copy-candidate.sha256.8dbe752a1d028864fc568fbc7ee6f3bc699c87367a7afd67809bc587a15085f3",
      artifactId:
        "bg.artifact.cpp-cute-frontend.sha256.e7b688cd0044c6ca9052f5dc555dc0ff8597ee3f5a6597d99a451f7aa00db5b7",
      authorizationId:
        "bg.cpp.browser-view-copy-authorization.sha256.2dd020ac49f1c518f3b0569e1d11d8164967f7306b0c94e1784c0d6dcdb76878",
      executionEvidenceId:
        "bg.cpp.browser-worker-execution.sha256.51065b90c0b302cc98ebf6f22d1b1e161977ba1b0eb68b8a898291e6f4305e4d",
      layoutSemanticHash:
        "c2b5e8a0489bd2ee5a54d15399af95b91d9fe102aab63e450361500ffa946a6f",
      kernelSemanticHash:
        "e335ea9d9e9a38f591c80c737b8a33401578739e02d6892e5f1907e6b76e6ff2",
      cpuDestinationHash:
        "1d1c9094fa8fb4c5a2b9f83093cd1ccca1ec8402de26abe1dee6c3843e7d5c2a",
      webGpuDestinationHash:
        "1d1c9094fa8fb4c5a2b9f83093cd1ccca1ec8402de26abe1dee6c3843e7d5c2a",
      deviceProfileHash:
        "823ff00eae2f4ee770ad55563920a43566dd904664915b855f7728c99c6e625f",
    },
    {
      caseId: "rank1",
      evidenceId:
        "bg.cpp.browser-exact-distribution-case-convergence.sha256.11ef1426b73046965598bceaca4450c738282bd28fe75eafde5a6cff389170bc",
      sourceSha256:
        "7c8fc9f261fab7181e9c25d124f2604d31a48a5c9bc49e043c42f9371a27c1c7",
      dtype: "f32",
      coordinateRank: 1,
      candidateId:
        "bg.cpp.browser-worker-view-copy-candidate.sha256.01e30bca3f0ed233aa382d6aa0d4f8e448b52f04fc059aea86f98c65d02050c1",
      artifactId:
        "bg.artifact.cpp-cute-frontend.sha256.c1d34501b96eab0d2c998dc73938f8753cbc3b28fe63e2d63e587efe73bca4a3",
      authorizationId:
        "bg.cpp.browser-view-copy-authorization.sha256.dc1298fee92f782bf313ee0f58a214d47cc8dc62097f938087d34a7f033ab222",
      executionEvidenceId:
        "bg.cpp.browser-worker-execution.sha256.8d4e8996e5786032b7dc922228ba6b59e7177ecb67c0c09b8177ce96e7f641dd",
      layoutSemanticHash:
        "89bf81de47ee52bef3e927ce3ace9fb2d77296eb8214f5a4f0aad184b5816e43",
      kernelSemanticHash:
        "02f310f29ae7d8080c5a3f6926ae6ef0c65d32040b7cc8190db56ea40763931c",
      cpuDestinationHash:
        "5c6ab95d110f933c4112c8a9b1ea7451afb09a8e018aa1cf115b2da3cc4cd021",
      webGpuDestinationHash:
        "5c6ab95d110f933c4112c8a9b1ea7451afb09a8e018aa1cf115b2da3cc4cd021",
      deviceProfileHash:
        "823ff00eae2f4ee770ad55563920a43566dd904664915b855f7728c99c6e625f",
    },
    {
      caseId: "rank4",
      evidenceId:
        "bg.cpp.browser-exact-distribution-case-convergence.sha256.c1edef9806699f51160aa7445da94657ded8b134d07796f7f491c396c2af82a9",
      sourceSha256:
        "28d6094af10254112f25ea717739c836c2c74a4c3f35c7b88dcc45ad60e5a05a",
      dtype: "f32",
      coordinateRank: 4,
      candidateId:
        "bg.cpp.browser-worker-view-copy-candidate.sha256.e8f0a8f583ab12d91187041592289a801ce0fa5043fc6650babd823c4113fe99",
      artifactId:
        "bg.artifact.cpp-cute-frontend.sha256.9b2dcfe7005005351bd6067d350fb0239e13950c47b72b931d39ff0ab6906212",
      authorizationId:
        "bg.cpp.browser-view-copy-authorization.sha256.d0689f715599b965f3a0f3061baf0c1de1369bfc6f41ad589b6e59a8e8106a65",
      executionEvidenceId:
        "bg.cpp.browser-worker-execution.sha256.7ac10b429364ec1fec25526455b888f16170d268aecbd1c45d4dfbb5b18d2b3c",
      layoutSemanticHash:
        "f71fa705c81324cbb001d7bcf06b48228b5eabcbad461ab99c82f138b59c4b41",
      kernelSemanticHash:
        "eeafc342c45e9c0046e82b67fb9e5f1ad9b991beae24063e2a88ca732dc2700b",
      cpuDestinationHash:
        "05c5aefa8646898996289d943056e68b1844190feaa58bbef23d5690ff72828d",
      webGpuDestinationHash:
        "05c5aefa8646898996289d943056e68b1844190feaa58bbef23d5690ff72828d",
      deviceProfileHash:
        "823ff00eae2f4ee770ad55563920a43566dd904664915b855f7728c99c6e625f",
    },
    {
      caseId: "strided-slice",
      evidenceId:
        "bg.cpp.browser-exact-distribution-case-convergence.sha256.277c24dd13251cc871ba4c4d99c2c43505036fcaaf8d4eefe7b942908c467ee5",
      sourceSha256:
        "55f4f5fcf55093a05cb977e3b83479098f6ddc42b830ec63f44b97f27fe3264a",
      dtype: "f32",
      coordinateRank: 2,
      candidateId:
        "bg.cpp.browser-worker-view-copy-candidate.sha256.85bc2de4a493533094275a9337633305083d64bc59223c6f7bb09e746724779d",
      artifactId:
        "bg.artifact.cpp-cute-frontend.sha256.37e15e466ad96cc0dd99fe779101836d5b8072cff94595963880bc48a84bbf94",
      authorizationId:
        "bg.cpp.browser-view-copy-authorization.sha256.b9e0ff1306d08cef94edd4bbe926e13c68935599957e42ed74ed3ff25f32687b",
      executionEvidenceId:
        "bg.cpp.browser-worker-execution.sha256.ba1208d5282027c62703c83b0bf26869db92b2e87155051ef7b092898e50e9e4",
      layoutSemanticHash:
        "c43e46d9b4a2fe623b70c5ff67ce5b35e260b5a498a2ede0bf67984d1cc46399",
      kernelSemanticHash:
        "939d9e9ec1125485b71ac28aa5bf491928c0721e7642a72d6dc18e6734187d73",
      cpuDestinationHash:
        "318df11b8031b82c60a6971b1b1172e68aa23bf1c7edded1135392783c961fd4",
      webGpuDestinationHash:
        "318df11b8031b82c60a6971b1b1172e68aa23bf1c7edded1135392783c961fd4",
      deviceProfileHash:
        "823ff00eae2f4ee770ad55563920a43566dd904664915b855f7728c99c6e625f",
    },
    {
      caseId: "broadcast",
      evidenceId:
        "bg.cpp.browser-exact-distribution-case-convergence.sha256.cfa9c372bfcc7b470cff5ad236cd5dcb356ee08368bc81bfa03614c57812d872",
      sourceSha256:
        "bfd91bdaac57ef7314570a8de56f26165a7b263593f319d728c53c13ef7c6376",
      dtype: "f32",
      coordinateRank: 2,
      candidateId:
        "bg.cpp.browser-worker-view-copy-candidate.sha256.5021bdc1b31270a6344206c6fbb1c72fed7f1846f100787fc5c52c2bb5bc6536",
      artifactId:
        "bg.artifact.cpp-cute-frontend.sha256.5b844fdd341a1b332bedd59396b4870881070a582a8b1d5848806495738cea0a",
      authorizationId:
        "bg.cpp.browser-view-copy-authorization.sha256.f29a1592ea6fcedc739455e3e361782603c844901b016165dc89c8e13e8d8b9b",
      executionEvidenceId:
        "bg.cpp.browser-worker-execution.sha256.cc9b753bf0a338fb2dc4089e84ec5a0c6692065e50706e9f9348364fb8db5822",
      layoutSemanticHash:
        "5e72e11b9dc901e8c9f4cc05221c3f24fced5f2701bd129aa3dd2c183631a718",
      kernelSemanticHash:
        "bfa123e13a4d162ee02a736490494d5ee5d3577d875c819283f0f2243c5fea16",
      cpuDestinationHash:
        "60d8fbebb1cf3026189c500e30ae88802f7e3d91651d6fb40774aff808ccfb1b",
      webGpuDestinationHash:
        "60d8fbebb1cf3026189c500e30ae88802f7e3d91651d6fb40774aff808ccfb1b",
      deviceProfileHash:
        "823ff00eae2f4ee770ad55563920a43566dd904664915b855f7728c99c6e625f",
    },
    {
      caseId: "i32-rank2",
      evidenceId:
        "bg.cpp.browser-exact-distribution-case-convergence.sha256.71949cf369448d900588688c29bce20b98b3ddc9bf22104707c24cff58365dae",
      sourceSha256:
        "88a083b141a5b7a85a9a1f2420873029f891cc1738e74ababe849a58bb839577",
      dtype: "i32",
      coordinateRank: 2,
      candidateId:
        "bg.cpp.browser-worker-view-copy-candidate.sha256.70850398376f214a403d8eab407ff783c73bd501a716b747dd68c56d66552682",
      artifactId:
        "bg.artifact.cpp-cute-frontend.sha256.95b28c81deffed79cacbb48236a119a2b6eb77e6ccd1e133119063154bcc6e52",
      authorizationId:
        "bg.cpp.browser-view-copy-authorization.sha256.aee3957d54999f5a5689ea3e475498b6264bb959e396c250ba21f88179a50adb",
      executionEvidenceId:
        "bg.cpp.browser-worker-execution.sha256.bb8777f845865944e72b8aa92b14ad37bf7119ae217c83632a8ca1c03f5ecad4",
      layoutSemanticHash:
        "a29489aaa6c39240c1b5082e3b3b4cf89e7ecdf3a645a25380955799fa884210",
      kernelSemanticHash:
        "bfb93603b68d0b6f96785f415bb17398767d4b82d3204b25a04d0fbd2500201e",
      cpuDestinationHash:
        "e9eb93a4e037911ffa699e1e8fc936d31539ae65fc9a071b36aee9531757655d",
      webGpuDestinationHash:
        "e9eb93a4e037911ffa699e1e8fc936d31539ae65fc9a071b36aee9531757655d",
      deviceProfileHash:
        "823ff00eae2f4ee770ad55563920a43566dd904664915b855f7728c99c6e625f",
    },
    {
      caseId: "u32-broadcast",
      evidenceId:
        "bg.cpp.browser-exact-distribution-case-convergence.sha256.71139a57ec9f39425668b331d7d497b38bf1003796bff0822ea4e93d9bec9249",
      sourceSha256:
        "c21158f1cb394377b5b8057435bfaad64b515689a8b04391a1bcae643a567536",
      dtype: "u32",
      coordinateRank: 2,
      candidateId:
        "bg.cpp.browser-worker-view-copy-candidate.sha256.3b3ffa0ff6bd42b70b0bfbed200d6c9b4aedbfa125b4a7e0ad6558cb85117b57",
      artifactId:
        "bg.artifact.cpp-cute-frontend.sha256.83a0283b156948f9ce003e46a9212cd5e3011796c28d9e2ea6717cff2d7d6e3d",
      authorizationId:
        "bg.cpp.browser-view-copy-authorization.sha256.dd0686aae3b332cdb6e47bea26441ae4a2e7e16a320bebc32947c9fee515d523",
      executionEvidenceId:
        "bg.cpp.browser-worker-execution.sha256.d4901dfbfd84daf746f0714d5ce2beba13e024dff8b43a08d042d74508ce7c3b",
      layoutSemanticHash:
        "063922fecf9239305bf20d810b49e979307994e2695a68ddb2c9913f08fb5f4e",
      kernelSemanticHash:
        "72298c98ee1bef581aa04cf43a7ad859be3393396852df52075e1771043d0775",
      cpuDestinationHash:
        "11bd283413d5dc57742c673834ce3605098c3ecd042b8782501b43f2a3aad300",
      webGpuDestinationHash:
        "11bd283413d5dc57742c673834ce3605098c3ecd042b8782501b43f2a3aad300",
      deviceProfileHash:
        "823ff00eae2f4ee770ad55563920a43566dd904664915b855f7728c99c6e625f",
    },
  ],
  claims: {
    backendExecutionAuthorityMinted: false,
    completeDestinationBitComparisonPassedForEveryCase: true,
    cpuReferenceConvergenceObservedForEveryCase: true,
    distributionAuthorized: false,
    exactCandidatesAuthorizedThroughSharedSeam: true,
    exactEightCaseBrowserWorkerCompilationObserved: true,
    exactPrivateDistributionTreeVerified: true,
    externalProducerTrusted: false,
    licenseReviewComplete: false,
    localEngineeringProducerAuthenticated: true,
    nonzeroOffsetCanariesPreservedForEveryCase: true,
    packagePinnedFullDistributionReproducibilityMatched: true,
    releaseReady: false,
    requiredRealWebGpuConvergenceObservedForEveryCase: true,
  },
} as const satisfies JsonObject;

export type CppCuteBrowserExactDistributionConvergenceV1Resource =
  typeof VALUE;

export const CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_V1_RESOURCE =
  deepFreezeJson(VALUE) as unknown as
    CppCuteBrowserExactDistributionConvergenceV1Resource;
