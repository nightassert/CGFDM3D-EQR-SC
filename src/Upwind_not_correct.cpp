
#ifdef UW
    float u_phy[9], u_phy_T_n[9], u_phy_T_p[9];
    float Riemann_flux_T[9], Riemann_flux_phy[9];
    float nx, ny, nz, sx, sy, sz, tx, ty, tz;
    float delta_tau_nn, delta_tau_ns, delta_tau_nt, delta_v_n, delta_v_s, delta_v_t;
    float mu_p, lambda_p, buoyancy_p;
    float cp, cs, cp_p, cs_p;
    float rho, rho_p;
#endif

#ifdef UW

    mu_p = CJM[idx_p1 * CJMSIZE + 10];
    lambda_p = CJM[idx_p1 * CJMSIZE + 11];
    buoyancy_p = CJM[idx_p1 * CJMSIZE + 12];
    buoyancy_p *= Crho;

    rho = 1.0 / buoyancy;
    rho_p = 1.0 / buoyancy_p;

    cp = sqrt(mu / rho);
    cs = sqrt((lambda + 2 * mu) / rho);
    cp_p = sqrt(mu_p / rho_p);
    cs_p = sqrt((lambda_p + 2 * mu_p) / rho_p);

    nx = CJM[idx * CJMSIZE + 13];
    ny = CJM[idx * CJMSIZE + 14];
    nz = CJM[idx * CJMSIZE + 15];
    sx = CJM[idx * CJMSIZE + 16];
    sy = CJM[idx * CJMSIZE + 17];
    sz = CJM[idx * CJMSIZE + 18];
    tx = CJM[idx * CJMSIZE + 19];
    ty = CJM[idx * CJMSIZE + 20];
    tz = CJM[idx * CJMSIZE + 21];

    // Calculate physical variables
    u_phy[0] = lambda * u_ip12n[1] + lambda * u_ip12n[2] + u_ip12n[0] * (lambda + 2 * mu);
    u_phy[1] = lambda * u_ip12n[0] + lambda * u_ip12n[2] + u_ip12n[1] * (lambda + 2 * mu);
    u_phy[2] = lambda * u_ip12n[0] + lambda * u_ip12n[1] + u_ip12n[2] * (lambda + 2 * mu);
    u_phy[3] = 2 * mu * u_ip12n[3];
    u_phy[4] = 2 * mu * u_ip12n[5];
    u_phy[5] = 2 * mu * u_ip12n[4];
    u_phy[6] = u_ip12n[6] * buoyancy;
    u_phy[7] = u_ip12n[7] * buoyancy;
    u_phy[8] = u_ip12n[8] * buoyancy;

    // Rotate physical variables
    u_phy_T_n[0] = (u_phy[2] * (sx * sx) * (ty * ty) - 2 * u_phy[5] * (sx * sx) * ty * tz + u_phy[1] * (sx * sx) * (tz * tz) - 2 * u_phy[2] * sx * sy * tx * ty + 2 * u_phy[5] * sx * sy * tx * tz + 2 * u_phy[4] * sx * sy * ty * tz - 2 * u_phy[3] * sx * sy * (tz * tz) + 2 * u_phy[5] * sx * sz * tx * ty - 2 * u_phy[1] * sx * sz * tx * tz - 2 * u_phy[4] * sx * sz * (ty * ty) + 2 * u_phy[3] * sx * sz * ty * tz + u_phy[2] * (sy * sy) * (tx * tx) - 2 * u_phy[4] * (sy * sy) * tx * tz + u_phy[0] * (sy * sy) * (tz * tz) - 2 * u_phy[5] * sy * sz * (tx * tx) + 2 * u_phy[4] * sy * sz * tx * ty + 2 * u_phy[3] * sy * sz * tx * tz - 2 * u_phy[0] * sy * sz * ty * tz + u_phy[1] * (sz * sz) * (tx * tx) - 2 * u_phy[3] * (sz * sz) * tx * ty + u_phy[0] * (sz * sz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    u_phy_T_n[1] = (u_phy[2] * (nx * nx) * (ty * ty) - 2 * u_phy[5] * (nx * nx) * ty * tz + u_phy[1] * (nx * nx) * (tz * tz) - 2 * u_phy[2] * nx * ny * tx * ty + 2 * u_phy[5] * nx * ny * tx * tz + 2 * u_phy[4] * nx * ny * ty * tz - 2 * u_phy[3] * nx * ny * (tz * tz) + 2 * u_phy[5] * nx * nz * tx * ty - 2 * u_phy[1] * nx * nz * tx * tz - 2 * u_phy[4] * nx * nz * (ty * ty) + 2 * u_phy[3] * nx * nz * ty * tz + u_phy[2] * (ny * ny) * (tx * tx) - 2 * u_phy[4] * (ny * ny) * tx * tz + u_phy[0] * (ny * ny) * (tz * tz) - 2 * u_phy[5] * ny * nz * (tx * tx) + 2 * u_phy[4] * ny * nz * tx * ty + 2 * u_phy[3] * ny * nz * tx * tz - 2 * u_phy[0] * ny * nz * ty * tz + u_phy[1] * (nz * nz) * (tx * tx) - 2 * u_phy[3] * (nz * nz) * tx * ty + u_phy[0] * (nz * nz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    u_phy_T_n[2] = (u_phy[2] * (nx * nx) * (sy * sy) - 2 * u_phy[5] * (nx * nx) * sy * sz + u_phy[1] * (nx * nx) * (sz * sz) - 2 * u_phy[2] * nx * ny * sx * sy + 2 * u_phy[5] * nx * ny * sx * sz + 2 * u_phy[4] * nx * ny * sy * sz - 2 * u_phy[3] * nx * ny * (sz * sz) + 2 * u_phy[5] * nx * nz * sx * sy - 2 * u_phy[1] * nx * nz * sx * sz - 2 * u_phy[4] * nx * nz * (sy * sy) + 2 * u_phy[3] * nx * nz * sy * sz + u_phy[2] * (ny * ny) * (sx * sx) - 2 * u_phy[4] * (ny * ny) * sx * sz + u_phy[0] * (ny * ny) * (sz * sz) - 2 * u_phy[5] * ny * nz * (sx * sx) + 2 * u_phy[4] * ny * nz * sx * sy + 2 * u_phy[3] * ny * nz * sx * sz - 2 * u_phy[0] * ny * nz * sy * sz + u_phy[1] * (nz * nz) * (sx * sx) - 2 * u_phy[3] * (nz * nz) * sx * sy + u_phy[0] * (nz * nz) * (sy * sy)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    u_phy_T_n[3] = -(nx * sx * (ty * ty) * u_phy[2] + nx * sx * (tz * tz) * u_phy[1] + ny * sy * (tx * tx) * u_phy[2] - nx * sy * (tz * tz) * u_phy[3] - ny * sx * (tz * tz) * u_phy[3] - nx * sz * (ty * ty) * u_phy[4] - nz * sx * (ty * ty) * u_phy[4] - ny * sz * (tx * tx) * u_phy[5] - nz * sy * (tx * tx) * u_phy[5] + ny * sy * (tz * tz) * u_phy[0] + nz * sz * (tx * tx) * u_phy[1] + nz * sz * (ty * ty) * u_phy[0] - nx * sy * tx * ty * u_phy[2] - ny * sx * tx * ty * u_phy[2] - 2 * nx * sx * ty * tz * u_phy[5] + nx * sy * tx * tz * u_phy[5] + nx * sz * tx * ty * u_phy[5] + ny * sx * tx * tz * u_phy[5] + nz * sx * tx * ty * u_phy[5] - nx * sz * tx * tz * u_phy[1] - nz * sx * tx * tz * u_phy[1] + nx * sy * ty * tz * u_phy[4] + ny * sx * ty * tz * u_phy[4] - 2 * ny * sy * tx * tz * u_phy[4] + ny * sz * tx * ty * u_phy[4] + nz * sy * tx * ty * u_phy[4] + nx * sz * ty * tz * u_phy[3] + ny * sz * tx * tz * u_phy[3] + nz * sx * ty * tz * u_phy[3] + nz * sy * tx * tz * u_phy[3] - 2 * nz * sz * tx * ty * u_phy[3] - ny * sz * ty * tz * u_phy[0] - nz * sy * ty * tz * u_phy[0]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    u_phy_T_n[4] = -(nx * (sy * sy) * tx * u_phy[2] + nx * (sz * sz) * tx * u_phy[1] + ny * (sx * sx) * ty * u_phy[2] - nx * (sz * sz) * ty * u_phy[3] - ny * (sz * sz) * tx * u_phy[3] - nx * (sy * sy) * tz * u_phy[4] - nz * (sy * sy) * tx * u_phy[4] - ny * (sx * sx) * tz * u_phy[5] - nz * (sx * sx) * ty * u_phy[5] + ny * (sz * sz) * ty * u_phy[0] + nz * (sx * sx) * tz * u_phy[1] + nz * (sy * sy) * tz * u_phy[0] - nx * sx * sy * ty * u_phy[2] - ny * sx * sy * tx * u_phy[2] + nx * sx * sy * tz * u_phy[5] + nx * sx * sz * ty * u_phy[5] - 2 * nx * sy * sz * tx * u_phy[5] + ny * sx * sz * tx * u_phy[5] + nz * sx * sy * tx * u_phy[5] - nx * sx * sz * tz * u_phy[1] - nz * sx * sz * tx * u_phy[1] + nx * sy * sz * ty * u_phy[4] + ny * sx * sy * tz * u_phy[4] - 2 * ny * sx * sz * ty * u_phy[4] + ny * sy * sz * tx * u_phy[4] + nz * sx * sy * ty * u_phy[4] + nx * sy * sz * tz * u_phy[3] + ny * sx * sz * tz * u_phy[3] - 2 * nz * sx * sy * tz * u_phy[3] + nz * sx * sz * ty * u_phy[3] + nz * sy * sz * tx * u_phy[3] - ny * sy * sz * tz * u_phy[0] - nz * sy * sz * ty * u_phy[0]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    u_phy_T_n[5] = -((ny * ny) * sx * tx * u_phy[2] + (nz * nz) * sx * tx * u_phy[1] + (nx * nx) * sy * ty * u_phy[2] - (nz * nz) * sx * ty * u_phy[3] - (nz * nz) * sy * tx * u_phy[3] - (ny * ny) * sx * tz * u_phy[4] - (ny * ny) * sz * tx * u_phy[4] - (nx * nx) * sy * tz * u_phy[5] - (nx * nx) * sz * ty * u_phy[5] + (nz * nz) * sy * ty * u_phy[0] + (nx * nx) * sz * tz * u_phy[1] + (ny * ny) * sz * tz * u_phy[0] - nx * ny * sx * ty * u_phy[2] - nx * ny * sy * tx * u_phy[2] + nx * ny * sx * tz * u_phy[5] + nx * ny * sz * tx * u_phy[5] + nx * nz * sx * ty * u_phy[5] + nx * nz * sy * tx * u_phy[5] - 2 * ny * nz * sx * tx * u_phy[5] - nx * nz * sx * tz * u_phy[1] - nx * nz * sz * tx * u_phy[1] + nx * ny * sy * tz * u_phy[4] + nx * ny * sz * ty * u_phy[4] - 2 * nx * nz * sy * ty * u_phy[4] + ny * nz * sx * ty * u_phy[4] + ny * nz * sy * tx * u_phy[4] - 2 * nx * ny * sz * tz * u_phy[3] + nx * nz * sy * tz * u_phy[3] + nx * nz * sz * ty * u_phy[3] + ny * nz * sx * tz * u_phy[3] + ny * nz * sz * tx * u_phy[3] - ny * nz * sy * tz * u_phy[0] - ny * nz * sz * ty * u_phy[0]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    u_phy_T_n[6] = (sx * ty * u_phy[8] - sy * tx * u_phy[8] - sx * tz * u_phy[7] + sz * tx * u_phy[7] + sy * tz * u_phy[6] - sz * ty * u_phy[6]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    u_phy_T_n[7] = -(nx * ty * u_phy[8] - ny * tx * u_phy[8] - nx * tz * u_phy[7] + nz * tx * u_phy[7] + ny * tz * u_phy[6] - nz * ty * u_phy[6]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    u_phy_T_n[8] = (nx * sy * u_phy[8] - ny * sx * u_phy[8] - nx * sz * u_phy[7] + nz * sx * u_phy[7] + ny * sz * u_phy[6] - nz * sy * u_phy[6]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);

    // Calculate physical variables
    u_phy[0] = lambda * u_ip12p[1] + lambda * u_ip12p[2] + u_ip12p[0] * (lambda + 2 * mu);
    u_phy[1] = lambda * u_ip12p[0] + lambda * u_ip12p[2] + u_ip12p[1] * (lambda + 2 * mu);
    u_phy[2] = lambda * u_ip12p[0] + lambda * u_ip12p[1] + u_ip12p[2] * (lambda + 2 * mu);
    u_phy[3] = 2 * mu * u_ip12p[3];
    u_phy[4] = 2 * mu * u_ip12p[5];
    u_phy[5] = 2 * mu * u_ip12p[4];
    u_phy[6] = u_ip12p[6] * buoyancy;
    u_phy[7] = u_ip12p[7] * buoyancy;
    u_phy[8] = u_ip12p[8] * buoyancy;

    // Rotate physical variables
    u_phy_T_p[0] = (u_phy[2] * (sx * sx) * (ty * ty) - 2 * u_phy[5] * (sx * sx) * ty * tz + u_phy[1] * (sx * sx) * (tz * tz) - 2 * u_phy[2] * sx * sy * tx * ty + 2 * u_phy[5] * sx * sy * tx * tz + 2 * u_phy[4] * sx * sy * ty * tz - 2 * u_phy[3] * sx * sy * (tz * tz) + 2 * u_phy[5] * sx * sz * tx * ty - 2 * u_phy[1] * sx * sz * tx * tz - 2 * u_phy[4] * sx * sz * (ty * ty) + 2 * u_phy[3] * sx * sz * ty * tz + u_phy[2] * (sy * sy) * (tx * tx) - 2 * u_phy[4] * (sy * sy) * tx * tz + u_phy[0] * (sy * sy) * (tz * tz) - 2 * u_phy[5] * sy * sz * (tx * tx) + 2 * u_phy[4] * sy * sz * tx * ty + 2 * u_phy[3] * sy * sz * tx * tz - 2 * u_phy[0] * sy * sz * ty * tz + u_phy[1] * (sz * sz) * (tx * tx) - 2 * u_phy[3] * (sz * sz) * tx * ty + u_phy[0] * (sz * sz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    u_phy_T_p[1] = (u_phy[2] * (nx * nx) * (ty * ty) - 2 * u_phy[5] * (nx * nx) * ty * tz + u_phy[1] * (nx * nx) * (tz * tz) - 2 * u_phy[2] * nx * ny * tx * ty + 2 * u_phy[5] * nx * ny * tx * tz + 2 * u_phy[4] * nx * ny * ty * tz - 2 * u_phy[3] * nx * ny * (tz * tz) + 2 * u_phy[5] * nx * nz * tx * ty - 2 * u_phy[1] * nx * nz * tx * tz - 2 * u_phy[4] * nx * nz * (ty * ty) + 2 * u_phy[3] * nx * nz * ty * tz + u_phy[2] * (ny * ny) * (tx * tx) - 2 * u_phy[4] * (ny * ny) * tx * tz + u_phy[0] * (ny * ny) * (tz * tz) - 2 * u_phy[5] * ny * nz * (tx * tx) + 2 * u_phy[4] * ny * nz * tx * ty + 2 * u_phy[3] * ny * nz * tx * tz - 2 * u_phy[0] * ny * nz * ty * tz + u_phy[1] * (nz * nz) * (tx * tx) - 2 * u_phy[3] * (nz * nz) * tx * ty + u_phy[0] * (nz * nz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    u_phy_T_p[2] = (u_phy[2] * (nx * nx) * (sy * sy) - 2 * u_phy[5] * (nx * nx) * sy * sz + u_phy[1] * (nx * nx) * (sz * sz) - 2 * u_phy[2] * nx * ny * sx * sy + 2 * u_phy[5] * nx * ny * sx * sz + 2 * u_phy[4] * nx * ny * sy * sz - 2 * u_phy[3] * nx * ny * (sz * sz) + 2 * u_phy[5] * nx * nz * sx * sy - 2 * u_phy[1] * nx * nz * sx * sz - 2 * u_phy[4] * nx * nz * (sy * sy) + 2 * u_phy[3] * nx * nz * sy * sz + u_phy[2] * (ny * ny) * (sx * sx) - 2 * u_phy[4] * (ny * ny) * sx * sz + u_phy[0] * (ny * ny) * (sz * sz) - 2 * u_phy[5] * ny * nz * (sx * sx) + 2 * u_phy[4] * ny * nz * sx * sy + 2 * u_phy[3] * ny * nz * sx * sz - 2 * u_phy[0] * ny * nz * sy * sz + u_phy[1] * (nz * nz) * (sx * sx) - 2 * u_phy[3] * (nz * nz) * sx * sy + u_phy[0] * (nz * nz) * (sy * sy)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    u_phy_T_p[3] = -(nx * sx * (ty * ty) * u_phy[2] + nx * sx * (tz * tz) * u_phy[1] + ny * sy * (tx * tx) * u_phy[2] - nx * sy * (tz * tz) * u_phy[3] - ny * sx * (tz * tz) * u_phy[3] - nx * sz * (ty * ty) * u_phy[4] - nz * sx * (ty * ty) * u_phy[4] - ny * sz * (tx * tx) * u_phy[5] - nz * sy * (tx * tx) * u_phy[5] + ny * sy * (tz * tz) * u_phy[0] + nz * sz * (tx * tx) * u_phy[1] + nz * sz * (ty * ty) * u_phy[0] - nx * sy * tx * ty * u_phy[2] - ny * sx * tx * ty * u_phy[2] - 2 * nx * sx * ty * tz * u_phy[5] + nx * sy * tx * tz * u_phy[5] + nx * sz * tx * ty * u_phy[5] + ny * sx * tx * tz * u_phy[5] + nz * sx * tx * ty * u_phy[5] - nx * sz * tx * tz * u_phy[1] - nz * sx * tx * tz * u_phy[1] + nx * sy * ty * tz * u_phy[4] + ny * sx * ty * tz * u_phy[4] - 2 * ny * sy * tx * tz * u_phy[4] + ny * sz * tx * ty * u_phy[4] + nz * sy * tx * ty * u_phy[4] + nx * sz * ty * tz * u_phy[3] + ny * sz * tx * tz * u_phy[3] + nz * sx * ty * tz * u_phy[3] + nz * sy * tx * tz * u_phy[3] - 2 * nz * sz * tx * ty * u_phy[3] - ny * sz * ty * tz * u_phy[0] - nz * sy * ty * tz * u_phy[0]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    u_phy_T_p[4] = -(nx * (sy * sy) * tx * u_phy[2] + nx * (sz * sz) * tx * u_phy[1] + ny * (sx * sx) * ty * u_phy[2] - nx * (sz * sz) * ty * u_phy[3] - ny * (sz * sz) * tx * u_phy[3] - nx * (sy * sy) * tz * u_phy[4] - nz * (sy * sy) * tx * u_phy[4] - ny * (sx * sx) * tz * u_phy[5] - nz * (sx * sx) * ty * u_phy[5] + ny * (sz * sz) * ty * u_phy[0] + nz * (sx * sx) * tz * u_phy[1] + nz * (sy * sy) * tz * u_phy[0] - nx * sx * sy * ty * u_phy[2] - ny * sx * sy * tx * u_phy[2] + nx * sx * sy * tz * u_phy[5] + nx * sx * sz * ty * u_phy[5] - 2 * nx * sy * sz * tx * u_phy[5] + ny * sx * sz * tx * u_phy[5] + nz * sx * sy * tx * u_phy[5] - nx * sx * sz * tz * u_phy[1] - nz * sx * sz * tx * u_phy[1] + nx * sy * sz * ty * u_phy[4] + ny * sx * sy * tz * u_phy[4] - 2 * ny * sx * sz * ty * u_phy[4] + ny * sy * sz * tx * u_phy[4] + nz * sx * sy * ty * u_phy[4] + nx * sy * sz * tz * u_phy[3] + ny * sx * sz * tz * u_phy[3] - 2 * nz * sx * sy * tz * u_phy[3] + nz * sx * sz * ty * u_phy[3] + nz * sy * sz * tx * u_phy[3] - ny * sy * sz * tz * u_phy[0] - nz * sy * sz * ty * u_phy[0]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    u_phy_T_p[5] = -((ny * ny) * sx * tx * u_phy[2] + (nz * nz) * sx * tx * u_phy[1] + (nx * nx) * sy * ty * u_phy[2] - (nz * nz) * sx * ty * u_phy[3] - (nz * nz) * sy * tx * u_phy[3] - (ny * ny) * sx * tz * u_phy[4] - (ny * ny) * sz * tx * u_phy[4] - (nx * nx) * sy * tz * u_phy[5] - (nx * nx) * sz * ty * u_phy[5] + (nz * nz) * sy * ty * u_phy[0] + (nx * nx) * sz * tz * u_phy[1] + (ny * ny) * sz * tz * u_phy[0] - nx * ny * sx * ty * u_phy[2] - nx * ny * sy * tx * u_phy[2] + nx * ny * sx * tz * u_phy[5] + nx * ny * sz * tx * u_phy[5] + nx * nz * sx * ty * u_phy[5] + nx * nz * sy * tx * u_phy[5] - 2 * ny * nz * sx * tx * u_phy[5] - nx * nz * sx * tz * u_phy[1] - nx * nz * sz * tx * u_phy[1] + nx * ny * sy * tz * u_phy[4] + nx * ny * sz * ty * u_phy[4] - 2 * nx * nz * sy * ty * u_phy[4] + ny * nz * sx * ty * u_phy[4] + ny * nz * sy * tx * u_phy[4] - 2 * nx * ny * sz * tz * u_phy[3] + nx * nz * sy * tz * u_phy[3] + nx * nz * sz * ty * u_phy[3] + ny * nz * sx * tz * u_phy[3] + ny * nz * sz * tx * u_phy[3] - ny * nz * sy * tz * u_phy[0] - ny * nz * sz * ty * u_phy[0]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    u_phy_T_p[6] = (sx * ty * u_phy[8] - sy * tx * u_phy[8] - sx * tz * u_phy[7] + sz * tx * u_phy[7] + sy * tz * u_phy[6] - sz * ty * u_phy[6]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    u_phy_T_p[7] = -(nx * ty * u_phy[8] - ny * tx * u_phy[8] - nx * tz * u_phy[7] + nz * tx * u_phy[7] + ny * tz * u_phy[6] - nz * ty * u_phy[6]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    u_phy_T_p[8] = (nx * sy * u_phy[8] - ny * sx * u_phy[8] - nx * sz * u_phy[7] + nz * sx * u_phy[7] + ny * sz * u_phy[6] - nz * sy * u_phy[6]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);

    delta_tau_nn = u_phy_T_p[0] - u_phy_T_n[0];
    delta_tau_ns = u_phy_T_p[3] - u_phy_T_n[3];
    delta_tau_nt = u_phy_T_p[4] - u_phy_T_n[4];
    delta_v_n = u_phy_T_p[6] - u_phy_T_n[6];
    delta_v_s = u_phy_T_p[7] - u_phy_T_n[7];
    delta_v_t = u_phy_T_p[8] - u_phy_T_n[8];

    // R-H Riemann Solver
    Riemann_flux_T[0] = -(lambda + 2 * mu) * (delta_tau_nn + rho_p * cp_p * delta_v_n) / (rho * cp + rho_p * cp_p);
    Riemann_flux_T[1] = -(lambda) * (delta_tau_nn + rho_p * cp_p * delta_v_n) / (rho * cp + rho_p * cp_p);
    Riemann_flux_T[2] = -(lambda) * (delta_tau_nn + rho_p * cp_p * delta_v_n) / (rho * cp + rho_p * cp_p);
    Riemann_flux_T[3] = -(mu) * (delta_tau_ns + rho_p * cs_p * delta_v_s) / (rho * cs + rho_p * cs_p);
    Riemann_flux_T[4] = -(mu) * (delta_tau_nt + rho_p * cs_p * delta_v_t) / (rho * cs + rho_p * cs_p);
    Riemann_flux_T[5] = 0;
    Riemann_flux_T[6] = -(cp) * (delta_tau_nn + rho_p * cp_p * delta_v_n) / (rho * cp + rho_p * cp_p);
    Riemann_flux_T[7] = -(cs) * (delta_tau_ns + rho_p * cs_p * delta_v_s) / (rho * cs + rho_p * cs_p);
    Riemann_flux_T[8] = -(cs) * (delta_tau_nt + rho_p * cs_p * delta_v_t) / (rho * cs + rho_p * cs_p);

	// Rotate back physical variables
	Riemann_flux_phy[0] = Riemann_flux_T[0] * (nx * nx) + 2 * Riemann_flux_T[3] * nx * sx + 2 * Riemann_flux_T[4] * nx * tx + Riemann_flux_T[1] * (sx * sx) + 2 * Riemann_flux_T[5] * sx * tx + Riemann_flux_T[2] * (tx * tx);
	Riemann_flux_phy[1] = Riemann_flux_T[0] * (ny * ny) + 2 * Riemann_flux_T[3] * ny * sy + 2 * Riemann_flux_T[4] * ny * ty + Riemann_flux_T[1] * (sy * sy) + 2 * Riemann_flux_T[5] * sy * ty + Riemann_flux_T[2] * (ty * ty);
	Riemann_flux_phy[2] = Riemann_flux_T[0] * (nz * nz) + 2 * Riemann_flux_T[3] * nz * sz + 2 * Riemann_flux_T[4] * nz * tz + Riemann_flux_T[1] * (sz * sz) + 2 * Riemann_flux_T[5] * sz * tz + Riemann_flux_T[2] * (tz * tz);
	Riemann_flux_phy[3] = Riemann_flux_T[3] * (nx * sy + ny * sx) + Riemann_flux_T[4] * (nx * ty + ny * tx) + Riemann_flux_T[5] * (sx * ty + sy * tx) + nx * ny * Riemann_flux_T[0] + sx * sy * Riemann_flux_T[1] + tx * ty * Riemann_flux_T[2];
	Riemann_flux_phy[4] = Riemann_flux_T[3] * (nx * sz + nz * sx) + Riemann_flux_T[4] * (nx * tz + nz * tx) + Riemann_flux_T[5] * (sx * tz + sz * tx) + nx * nz * Riemann_flux_T[0] + sx * sz * Riemann_flux_T[1] + tx * tz * Riemann_flux_T[2];
	Riemann_flux_phy[5] = Riemann_flux_T[3] * (ny * sz + nz * sy) + Riemann_flux_T[4] * (ny * tz + nz * ty) + Riemann_flux_T[5] * (sy * tz + sz * ty) + ny * nz * Riemann_flux_T[0] + sy * sz * Riemann_flux_T[1] + ty * tz * Riemann_flux_T[2];
	Riemann_flux_phy[6] = nx * Riemann_flux_T[6] + sx * Riemann_flux_T[7] + tx * Riemann_flux_T[8];
	Riemann_flux_phy[7] = ny * Riemann_flux_T[6] + sy * Riemann_flux_T[7] + ty * Riemann_flux_T[8];
	Riemann_flux_phy[8] = nz * Riemann_flux_T[6] + sz * Riemann_flux_T[7] + tz * Riemann_flux_T[8];

	// Calculate conservative variables
	Riemann_flux[0] = (Riemann_flux_phy[0] * (lambda + mu)) / (2 * (mu * mu) + 3 * lambda * mu) - (lambda * Riemann_flux_phy[1]) / (2 * (2 * (mu * mu) + 3 * lambda * mu)) - (lambda * Riemann_flux_phy[2]) / (2 * (2 * (mu * mu) + 3 * lambda * mu));
	Riemann_flux[1] = (Riemann_flux_phy[1] * (lambda + mu)) / (2 * (mu * mu) + 3 * lambda * mu) - (lambda * Riemann_flux_phy[0]) / (2 * (2 * (mu * mu) + 3 * lambda * mu)) - (lambda * Riemann_flux_phy[2]) / (2 * (2 * (mu * mu) + 3 * lambda * mu));
	Riemann_flux[2] = (Riemann_flux_phy[2] * (lambda + mu)) / (2 * (mu * mu) + 3 * lambda * mu) - (lambda * Riemann_flux_phy[0]) / (2 * (2 * (mu * mu) + 3 * lambda * mu)) - (lambda * Riemann_flux_phy[1]) / (2 * (2 * (mu * mu) + 3 * lambda * mu));
	Riemann_flux[3] = Riemann_flux_phy[3] / (2 * mu);
	Riemann_flux[4] = Riemann_flux_phy[5] / (2 * mu);
	Riemann_flux[5] = Riemann_flux_phy[4] / (2 * mu);
	Riemann_flux[6] = Riemann_flux_phy[6] / buoyancy;
	Riemann_flux[7] = Riemann_flux_phy[7] / buoyancy;
	Riemann_flux[8] = Riemann_flux_phy[8] / buoyancy;
#endif