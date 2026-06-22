import { readCapabilityAlternatives, readCapabilityStringList, uniqueSorted } from "./assignment-profile.js";
import type {
  AssignmentCapabilityCatalog,
  AssignmentCapabilityCatalogAlternative,
  AssignmentCapabilityCatalogReference,
  AssignmentCapabilityEnvironment,
  AssignmentCapabilityEnvironmentInput,
  AssignmentCapabilityEvaluation,
  AssignmentCapabilityGateEvaluation,
  AssignmentCapabilityMode,
  AssignmentGateSpec,
  AssignmentProfile,
  AssignmentRunReadinessStatus,
} from "./assignment-types.js";

export function createAssignmentCapabilityEnvironment(
  input: AssignmentCapabilityEnvironmentInput = {},
): AssignmentCapabilityEnvironment {
  const browserCapabilities = uniqueSorted(input.browserCapabilities ?? []);
  const simulatedCapabilities = uniqueSorted(input.simulatedCapabilities ?? []);
  const externalCapabilities = uniqueSorted(input.externalCapabilities ?? []);
  const capabilities = uniqueSorted([
    ...browserCapabilities,
    ...simulatedCapabilities,
    ...externalCapabilities,
  ]);
  const capabilityModes: Record<string, AssignmentCapabilityMode> = {};

  for (const capability of externalCapabilities) {
    capabilityModes[capability] = "external";
  }
  for (const capability of simulatedCapabilities) {
    capabilityModes[capability] = "simulated";
  }
  for (const capability of browserCapabilities) {
    capabilityModes[capability] = "browser";
  }

  return { capabilities, capabilityModes };
}

export function evaluateAssignmentCapabilities(
  profile: AssignmentProfile,
  environment: AssignmentCapabilityEnvironment,
): AssignmentCapabilityEvaluation {
  const available = new Set(environment.capabilities);
  const capabilityModes = normalizeCapabilityModes(environment.capabilityModes);
  const gates = profile.gates
    .filter((gate) => gate.kind === "capability")
    .map((gate) => evaluateCapabilityGate(gate, available, capabilityModes));
  const satisfiedCapabilities = uniqueSorted(
    gates.flatMap((gate) => gate.satisfiedCapabilities),
  );

  return {
    ok: gates.every((gate) => gate.ok),
    satisfiedCapabilities,
    missingCapabilities: uniqueSorted(
      gates.flatMap((gate) => [
        ...gate.missingRequired,
        ...gate.missingAnyOf.flat(),
      ]),
    ),
    capabilityModes,
    gates,
  };
}

export function requiredAssignmentCapabilities(
  profile: AssignmentProfile,
): string[] {
  return uniqueSorted(
    profile.gates
      .filter((gate) => gate.kind === "capability")
      .flatMap((gate) => {
        const requires = readCapabilityStringList(gate.options.requires);
        const anyOf = readCapabilityAlternatives(gate.options.any_of);
        return [...requires, ...anyOf.flat()];
      }),
  );
}

export function createAssignmentCapabilityCatalog(
  profiles: readonly AssignmentProfile[],
): AssignmentCapabilityCatalog {
  const entries = new Map<string, MutableAssignmentCapabilityCatalogEntry>();

  for (const profile of profiles) {
    for (const gate of profile.gates) {
      if (gate.kind !== "capability") continue;
      for (const capability of readCapabilityStringList(gate.options.requires)) {
        getMutableCatalogEntry(entries, capability).requiredBy.push({
          profileId: profile.id,
          gate: gate.name,
        });
      }
      for (const group of readCapabilityAlternatives(gate.options.any_of)) {
        for (const capability of group) {
          getMutableCatalogEntry(entries, capability).alternativeIn.push({
            profileId: profile.id,
            gate: gate.name,
            group,
          });
        }
      }
    }
  }

  return {
    capabilities: [...entries.values()]
      .map((entry) => ({
        capability: entry.capability,
        profiles: uniqueSorted([
          ...entry.requiredBy.map((reference) => reference.profileId),
          ...entry.alternativeIn.map((reference) => reference.profileId),
        ]),
        requiredBy: sortCatalogReferences(entry.requiredBy),
        alternativeIn: sortCatalogAlternatives(entry.alternativeIn),
      }))
      .sort((a, b) => a.capability.localeCompare(b.capability)),
  };
}

interface MutableAssignmentCapabilityCatalogEntry {
  readonly capability: string;
  readonly requiredBy: AssignmentCapabilityCatalogReference[];
  readonly alternativeIn: AssignmentCapabilityCatalogAlternative[];
}

function getMutableCatalogEntry(
  entries: Map<string, MutableAssignmentCapabilityCatalogEntry>,
  capability: string,
): MutableAssignmentCapabilityCatalogEntry {
  let entry = entries.get(capability);
  if (!entry) {
    entry = { capability, requiredBy: [], alternativeIn: [] };
    entries.set(capability, entry);
  }
  return entry;
}

function sortCatalogReferences(
  references: readonly AssignmentCapabilityCatalogReference[],
): AssignmentCapabilityCatalogReference[] {
  return [...references].sort(compareCatalogReferences);
}

function sortCatalogAlternatives(
  alternatives: readonly AssignmentCapabilityCatalogAlternative[],
): AssignmentCapabilityCatalogAlternative[] {
  return [...alternatives].sort((a, b) =>
    compareCatalogReferences(a, b) ||
    a.group.join("\0").localeCompare(b.group.join("\0"))
  );
}

function compareCatalogReferences(
  a: AssignmentCapabilityCatalogReference,
  b: AssignmentCapabilityCatalogReference,
): number {
  return a.profileId.localeCompare(b.profileId) || a.gate.localeCompare(b.gate);
}

function evaluateCapabilityGate(
  gate: AssignmentGateSpec,
  available: ReadonlySet<string>,
  capabilityModes: Readonly<Record<string, AssignmentCapabilityMode>>,
): AssignmentCapabilityGateEvaluation {
  const requires = readCapabilityStringList(gate.options.requires);
  const anyOf = readCapabilityAlternatives(gate.options.any_of);
  const missingRequired = requires.filter((capability) => !available.has(capability));
  const satisfiedAnyOf = bestSatisfiedCapabilityGroup(anyOf, available, capabilityModes);
  const missingAnyOf =
    anyOf.length === 0 || satisfiedAnyOf.length > 0
      ? []
      : anyOf.map((group) => group.filter((capability) => !available.has(capability)));
  const message =
    typeof gate.options.message === "string" ? gate.options.message : undefined;
  const ok = missingRequired.length === 0 && missingAnyOf.length === 0;
  const selectedCapabilities = ok
    ? uniqueSorted([...requires, ...satisfiedAnyOf])
    : [];
  const satisfiedCapabilities = ok
    ? selectedCapabilities
    : uniqueSorted(requires.filter((capability) => available.has(capability)));

  return {
    name: gate.name,
    ok,
    status: capabilityGateStatus(selectedCapabilities, capabilityModes, ok),
    requires,
    anyOf,
    selectedAnyOf: ok ? satisfiedAnyOf : [],
    selectedCapabilities,
    satisfiedCapabilities,
    missingRequired,
    missingAnyOf,
    ...(message ? { message } : {}),
  };
}

function capabilityGateStatus(
  selectedCapabilities: readonly string[],
  capabilityModes: Readonly<Record<string, AssignmentCapabilityMode>>,
  ok: boolean,
): AssignmentRunReadinessStatus {
  if (!ok) return "blocked";
  if (selectedCapabilities.some((capability) => capabilityModes[capability] === "external")) {
    return "external-only";
  }
  if (selectedCapabilities.some((capability) => capabilityModes[capability] === "simulated")) {
    return "simulated";
  }
  return "runnable";
}

function bestSatisfiedCapabilityGroup(
  groups: readonly (readonly string[])[],
  available: ReadonlySet<string>,
  capabilityModes: Readonly<Record<string, AssignmentCapabilityMode>>,
): readonly string[] {
  return groups
    .filter((group) => group.every((capability) => available.has(capability)))
    .sort((a, b) => {
      const modeDiff =
        capabilityGroupModeRank(a, capabilityModes) -
        capabilityGroupModeRank(b, capabilityModes);
      if (modeDiff !== 0) return modeDiff;
      return a.length - b.length;
    })[0] ?? [];
}

function capabilityGroupModeRank(
  group: readonly string[],
  capabilityModes: Readonly<Record<string, AssignmentCapabilityMode>>,
): number {
  return Math.max(
    0,
    ...group.map((capability) => capabilityModeRank(capabilityModes[capability])),
  );
}

function capabilityModeRank(mode: AssignmentCapabilityMode | undefined): number {
  if (mode === "external") return 2;
  if (mode === "simulated") return 1;
  return 0;
}

function normalizeCapabilityModes(
  value: AssignmentCapabilityEnvironment["capabilityModes"],
): Readonly<Record<string, AssignmentCapabilityMode>> {
  if (!value) return {};
  const out: Record<string, AssignmentCapabilityMode> = {};
  for (const [capability, mode] of Object.entries(value)) {
    if (mode === "browser" || mode === "simulated" || mode === "external") {
      out[capability] = mode;
    }
  }
  return out;
}
