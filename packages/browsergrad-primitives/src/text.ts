import {
  createByteBpeOracle,
  createByteBpeOracleModule,
  decodeByteBpe,
  encodeByteBpe,
  GPT2_DEFAULT_SPECIAL_TOKENS,
  GPT2_PRETOKENIZER_PATTERN,
  createStreamingGate,
  deserializeByteBpeModel,
  serializeByteBpeModel,
  type ByteBpeModel,
  type ByteBpeOracleDefaults,
  type ByteBpeOracleModule,
  type SerializedByteBpeModel,
  type StreamingGate,
  type StreamingGateOptions,
  type TokenizerOracle,
  type TrainByteBpeOptions,
} from "@unlocalhosted/browsergrad-tokenizers";

export {
  createStreamingGate,
  decodeByteBpe,
  deserializeByteBpeModel,
  encodeByteBpe,
  GPT2_DEFAULT_SPECIAL_TOKENS,
  GPT2_PRETOKENIZER_PATTERN,
  serializeByteBpeModel,
  type ByteBpeModel,
  type ByteBpeOracleDefaults as ByteBpeReferenceDefaults,
  type ByteBpeOracleModule as ByteBpeReferenceModule,
  type SerializedByteBpeModel,
  type StreamingGate,
  type StreamingGateOptions,
  type TokenizerOracle as ByteBpeReference,
  type TrainByteBpeOptions,
};

const defaultByteBpeReference = createByteBpeOracle();

export const trainByteBpe = defaultByteBpeReference.trainByteBpe;

export function createByteBpeReference(
  defaults: ByteBpeOracleDefaults = {},
): TokenizerOracle {
  if (
    defaults.specialTokens === undefined &&
    defaults.pretokenizerPattern === undefined
  ) {
    return defaultByteBpeReference;
  }
  return createByteBpeOracle(defaults);
}

export function createByteBpeReferenceModule(
  defaults: ByteBpeOracleDefaults = {},
): ByteBpeOracleModule {
  return createByteBpeOracleModule(defaults);
}
