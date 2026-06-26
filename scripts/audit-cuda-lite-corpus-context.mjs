let auditHelpers;

export function setAuditHelpers(helpers) {
  auditHelpers = helpers;
}

export function requireAuditHelpers() {
  if (auditHelpers === undefined) throw new Error("cuda-lite corpus audit helpers were not initialized");
  return auditHelpers;
}
