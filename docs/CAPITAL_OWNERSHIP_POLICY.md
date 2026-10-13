## Capital Ownership Policy

### Scope

The V2 simulation environment uses virtual capital only.

### Real Kraken assets

All real Kraken holdings remain classified as `MANUAL_UNALLOCATED` in the real ownership ledger. They are not available to simulation, strategy evaluation, backtesting, paper trading, or simulated position calculations.

### Simulation isolation

Simulation state is independent from real-account operational state.

Simulation code must not:

- submit, edit, cancel, or construct exchange orders;
- import or invoke Kraken private API clients;
- modify `config/ownership_ledger.json`;
- modify `data/trading_safety_state.json`;
- modify reconciliation reports or account snapshots;
- use real account balances as starting simulation capital.

### Storage

Persistent simulation artifacts may be written only beneath:

`data/simulation/`

They must not be stored alongside real ownership, reconciliation, snapshot, or durable safety-state artifacts.

### Execution boundary

Simulation approval, simulated fills, virtual positions, and simulated performance are not authorization to trade. The repository remains read-only, and `trading_permitted` remains `False`.
