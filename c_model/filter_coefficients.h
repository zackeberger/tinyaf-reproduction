/* SciPy 1.16.0 butter(2, 40, fs=fs), sosfilt_zi; generated. */
typedef struct { int fs; TinyAFReal b0,b1,b2,a1,a2,z0,z1; } FilterCoefficients;
static const FilterCoefficients FILTERS[] = {
  {100, R(0x1.4723de2cf7935p-1), R(0x1.4723de2cf7935p+0), R(0x1.4723de2cf7935p-1), R(0x1.249a5ec65fe07p+0), R(0x1.a6b5764e3d18ep-2), R(0x1.71b843a610d95p-2), R(0x1.cf248c17641bap-3)},
  {128, R(0x1.ac330319bbb1cp-2), R(0x1.ac330319bbb1cp-1), R(0x1.ac330319bbb1cp-2), R(0x1.da0c6cfacca2fp-2), R(0x1.ad7f3ed84448dp-3), R(0x1.29e67e7322270p-1), R(0x1.aae6c75b331acp-3)},
  {200, R(0x1.a70f43c266255p-3), R(0x1.a70f43c266255p-2), R(0x1.a70f43c266255p-3), R(-0x1.7a656282cd09ep-2), R(0x1.9107d40f32a8fp-3), R(0x1.963c2f0f6676ap-1), R(0x1.6076fb3337c58p-7)},
  {250, R(0x1.299f9184a304bp-3), R(0x1.299f9184a304bp-2), R(0x1.299f9184a304bp-3), R(-0x1.57911fff1426cp-1), R(0x1.026163076e570p-2), R(0x1.b5981b9ed73ecp-1), R(-0x1.b646691473528p-4)},
  {256, R(0x1.1e987d242646ep-3), R(0x1.1e987d242646ep-2), R(0x1.1e987d242646ep-3), R(-0x1.66441016673bep-1), R(0x1.09b91a751b056p-2), R(0x1.b859e0b6f66e7p-1), R(-0x1.e9b36f8c1f881p-4)},
  {300, R(0x1.bc3361b5e54a7p-4), R(0x1.bc3361b5e54a7p-3), R(0x1.bc3361b5e54a7p-4), R(-0x1.c1299dc700c4bp-1), R(0x1.3e869d43e6d3fp-2), R(0x1.c87993c94356ap-1), R(-0x1.9ef389acdb029p-3)},
  {360, R(0x1.496a5191c117ep-4), R(0x1.496a5191c117ep-3), R(0x1.496a5191c117ep-4), R(-0x1.0da70799cacafp+0), R(0x1.80066ff8ec43cp-2), R(0x1.d6d2b5cdc7dcdp-1), R(-0x1.2dabdb947bfdap-2)},
  {500, R(0x1.79e966a97b7b3p-5), R(0x1.79e966a97b7b3p-4), R(0x1.79e966a97b7b3p-5), R(-0x1.4eaa3b4d7550ap+0), R(0x1.f79da08a93000p-2), R(0x1.e861699568489p-1), R(-0x1.c86073b56390ep-2)},
  {1000, R(0x1.b5c1179c34f3dp-7), R(0x1.b5c1179c34f3dp-6), R(0x1.b5c1179c34f3dp-7), R(-0x1.a5bfeff431c3dp+0), R(0x1.66dbf16226d6dp-1), R(0x1.f928fba18f2d2p-1), R(-0x1.6004ed03b603bp-1)},
};
