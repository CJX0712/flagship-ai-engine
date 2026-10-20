#!/usr/bin/env node
/**
 * 旗舰AI引擎 · 无头不变量自检入口
 * 期望输出：PASS 54/54 — 全部不变量自检通过
 */
import { runAllInvariants } from '../src/verify/invariants.js';

const t0 = Date.now();
const { results, passed, total, allOk } = runAllInvariants();
const ms = Date.now() - t0;

console.log('旗舰AI引擎 · 不变量自检\n' + '═'.repeat(64));
for (const r of results) {
  console.log(`${r.ok ? '✅' : '❌'} ${r.name}  (${r.ms}ms)`);
  for (const c of r.checks) {
    console.log(`   ${c.ok ? '✓' : '✗'} ${c.label}${c.info ? '  — ' + c.info : ''}`);
  }
}
console.log('─'.repeat(64));
for (const r of results.filter(r => !r.ok)) {
  console.log(`未通过套件：${r.name}`);
  for (const c of r.checks.filter(c => !c.ok)) {
    console.log(`  ✗ ${c.label}${c.info ? '  — ' + c.info : ''}`);
  }
}
console.log(allOk
  ? `PASS ${passed}/${total} — 全部不变量自检通过 (${ms}ms)`
  : `FAIL ${passed}/${total} — 存在未通过的不变量 (${ms}ms)`);
process.exit(allOk ? 0 : 1);
