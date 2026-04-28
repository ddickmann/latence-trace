-- latence-trace usage metering (D1 schema).
-- Applied via:  wrangler d1 migrations apply latence-usage

CREATE TABLE IF NOT EXISTS usage (
  tenant_id TEXT NOT NULL,
  month TEXT NOT NULL,          -- YYYY-MM
  band TEXT NOT NULL,           -- green / amber / red / unknown
  count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (tenant_id, month, band)
);

CREATE INDEX IF NOT EXISTS ix_usage_tenant_month
  ON usage(tenant_id, month);

CREATE TABLE IF NOT EXISTS tenants (
  tenant_id TEXT PRIMARY KEY,
  plan TEXT NOT NULL,           -- free / starter / business / enterprise
  rps_ceiling INTEGER NOT NULL,
  monthly_quota INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS billing_exports (
  tenant_id TEXT NOT NULL,
  month TEXT NOT NULL,
  total_count INTEGER NOT NULL,
  overage_count INTEGER NOT NULL DEFAULT 0,
  overage_rate_cents INTEGER NOT NULL DEFAULT 0,
  total_cents INTEGER NOT NULL DEFAULT 0,
  exported_to_stripe_at TEXT,
  stripe_invoice_id TEXT,
  PRIMARY KEY (tenant_id, month)
);
