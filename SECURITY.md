# Security and clinical-data handling

SynTrustBench runs locally and does not transmit input tables or telemetry. That does not make
every workflow safe by default.

- Do not commit identifiable or restricted real clinical data to this repository.
- Run inside the access-controlled environment authorized for the real data.
- Treat generated reports as potentially sensitive when subgroup counts or dataset names could
  disclose protected information.
- SHA-256 hashes identify exact inputs for reproducibility but do not anonymize a small or
  guessable file.
- Review output directories before sharing them.
- Clinical constraints are evaluated through a restricted expression parser; arbitrary function
  calls, imports, and attribute access are rejected.

The privacy module tests one stated empirical attacker. Passing it does not establish HIPAA
compliance, differential privacy, resistance to all attacks, or safe release for a particular
use. Formal privacy guarantees must come from the generator and its documented accounting.

Please report software vulnerabilities through a private GitHub security advisory when the
repository supports it. Scientific disagreements about metrics or thresholds should be filed as
regular issues and labeled as protocol discussions.
