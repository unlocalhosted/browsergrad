let normalizerHelpers;

export function setNormalizerHelpers(helpers) {
  normalizerHelpers = helpers;
}

export function requireNormalizerHelpers() {
  if (normalizerHelpers === undefined) throw new Error("cuda-lite source normalizer helpers were not initialized");
  return normalizerHelpers;
}
