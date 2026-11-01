/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: getParams.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-09-04
*   Discription: Get parameters from json file
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

#include "header.h"
#include <strings.h>

// "WENO-JS"/"WENO-Z" both run the alternative-flux WENO branch; the
// reconstruction is carried in a separate field.  "WENO" and "SCFDM" stay as
// aliases of WENO-JS so older params.json files keep working.
static SCHEME parseScheme(const char *name, WENO_RECONSTRUCTION *reconstruction)
{
	if (0 == strcasecmp(name, "WENO-Z") || 0 == strcasecmp(name, "WENOZ"))
	{
		*reconstruction = WENO_RECONSTRUCTION_Z;
		return SCHEME_WENO;
	}
	if (0 == strcasecmp(name, "WENO-JS") || 0 == strcasecmp(name, "WENO") ||
		0 == strcasecmp(name, "SCFDM"))
	{
		*reconstruction = WENO_RECONSTRUCTION_JS;
		return SCHEME_WENO;
	}
	if (0 == strcasecmp(name, "CGFDM"))
	{
		*reconstruction = WENO_RECONSTRUCTION_JS;
		return SCHEME_CGFDM;
	}

	printf("Unknown \"scheme\": %s. Use \"CGFDM\", \"WENO-JS\" or \"WENO-Z\".\n",
		   name);
	exit(101);
}

void getParams(PARAMS *params)
{
	memset(params, 0, sizeof(PARAMS));

	// Defaults.  Every key present in params.json overrides these values.
	params->scheme = SCHEME_CGFDM;
	params->IT_SKIP = 1;
	params->nPML = 10;
	// Historical behaviour: one global Vp_max for the LF dissipation, and the
	// Jiang & Shu reconstruction.
	params->wenoLocalAlpha = 0;
	params->wenoReconstruction = WENO_RECONSTRUCTION_JS;
	strcpy(params->OUT, "./output");

	char jsonFile[1024] = {0};
	strcpy(jsonFile, "params.json");
	FILE *fp;
	fp = fopen(jsonFile, "r");

	if (NULL == fp)
	{
		printf("There is not %s file!\n", jsonFile);
		exit(100);
	}

	fseek(fp, 0, SEEK_END);
	int len = ftell(fp);

	fseek(fp, 0, SEEK_SET);

	char *jsonStr = (char *)malloc(len + 1);

	if (NULL == jsonStr)
	{
		printf("Can't allocate json string memory\n");
		exit(102);
	}

	fread(jsonStr, sizeof(char), len, fp);
	jsonStr[len] = '\0';
	fclose(fp);

	cJSON *object, *item;
	object = cJSON_Parse(jsonStr);
	if (NULL == object)
	{
		printf("Can't parse json file!\n");
		exit(1001);
	}

	if ((item = cJSON_GetObjectItem(object, "scheme")) && item->valuestring)
		params->scheme = parseScheme(item->valuestring, &params->wenoReconstruction);

	if (item = cJSON_GetObjectItem(object, "TMAX"))
		params->TMAX = item->valuedouble;

	if (item = cJSON_GetObjectItem(object, "DT"))
		params->DT = item->valuedouble;

	if (item = cJSON_GetObjectItem(object, "DH"))
		params->DH = item->valuedouble;

	if (item = cJSON_GetObjectItem(object, "NX"))
		params->NX = item->valueint;
	if (item = cJSON_GetObjectItem(object, "NY"))
		params->NY = item->valueint;
	if (item = cJSON_GetObjectItem(object, "NZ"))
		params->NZ = item->valueint;

	if (item = cJSON_GetObjectItem(object, "PX"))
		params->PX = item->valueint;
	if (item = cJSON_GetObjectItem(object, "PY"))
		params->PY = item->valueint;
	if (item = cJSON_GetObjectItem(object, "PZ"))
		params->PZ = item->valueint;

	if (item = cJSON_GetObjectItem(object, "centerX"))
		params->centerX = item->valueint;
	if (item = cJSON_GetObjectItem(object, "centerY"))
		params->centerY = item->valueint;

	if (item = cJSON_GetObjectItem(object, "centerLongitude"))
		params->centerLongitude = item->valuedouble;
	if (item = cJSON_GetObjectItem(object, "centerLatitude"))
		params->centerLatitude = item->valuedouble;

	if (item = cJSON_GetObjectItem(object, "IT_SKIP"))
		params->IT_SKIP = item->valueint;

	if (item = cJSON_GetObjectItem(object, "sliceX"))
		params->sliceX = item->valueint;
	if (item = cJSON_GetObjectItem(object, "sliceY"))
		params->sliceY = item->valueint;
	if (item = cJSON_GetObjectItem(object, "sliceZ"))
		params->sliceZ = item->valueint;

	if (item = cJSON_GetObjectItem(object, "sliceFreeSurf"))
		params->sliceFreeSurf = item->valueint;

	if (item = cJSON_GetObjectItem(object, "nPML"))
		params->nPML = item->valueint;

	if (item = cJSON_GetObjectItem(object, "wenoLocalAlpha"))
		params->wenoLocalAlpha = item->valueint;

	if (item = cJSON_GetObjectItem(object, "gauss_hill"))
		params->gauss_hill = item->valueint;
	if (item = cJSON_GetObjectItem(object, "useTerrain"))
		params->useTerrain = item->valueint;
	if (item = cJSON_GetObjectItem(object, "useMedium"))
		params->useMedium = item->valueint;
	if (item = cJSON_GetObjectItem(object, "useMultiSource"))
		params->useMultiSource = item->valueint;

	if (item = cJSON_GetObjectItem(object, "LayeredModel"))
		params->LayeredModel = item->valueint;
	if (item = cJSON_GetObjectItem(object, "LayeredFileName"))
		strcpy(params->LayeredFileName, item->valuestring);

	if (item = cJSON_GetObjectItem(object, "useSingleSource(ricker)"))
		params->useSingleSource_ricker = item->valueint;

	if (item = cJSON_GetObjectItem(object, "useSingleSource(double_couple)"))
		params->useSingleSource_double_couple = item->valueint;

	if (item = cJSON_GetObjectItem(object, "strike"))
		params->strike = item->valuedouble;

	if (item = cJSON_GetObjectItem(object, "dip"))
		params->dip = item->valuedouble;

	if (item = cJSON_GetObjectItem(object, "rake"))
		params->rake = item->valuedouble;

	if (item = cJSON_GetObjectItem(object, "Mw"))
		params->Mw = item->valuedouble;

	if (item = cJSON_GetObjectItem(object, "duration"))
		params->duration = item->valuedouble;

	if (item = cJSON_GetObjectItem(object, "rickerfc"))
		params->rickerfc = item->valuedouble;

	if (item = cJSON_GetObjectItem(object, "ShenModel"))
		params->ShenModel = item->valueint;
	if (item = cJSON_GetObjectItem(object, "Crust_1Model"))
		params->Crust_1Model = item->valueint;

	if (item = cJSON_GetObjectItem(object, "itSlice"))
		params->itSlice = item->valueint;
	if (item = cJSON_GetObjectItem(object, "itStep"))
		params->itStep = item->valueint;

	if (item = cJSON_GetObjectItem(object, "waveOutput"))
		strcpy(params->waveOutput, item->valuestring);
	if (item = cJSON_GetObjectItem(object, "sliceName"))
		strcpy(params->sliceName, item->valuestring);

	if (item = cJSON_GetObjectItem(object, "itStart"))
		params->itStart = item->valueint;
	if (item = cJSON_GetObjectItem(object, "itEnd"))
		params->itEnd = item->valueint;

	if (item = cJSON_GetObjectItem(object, "igpu"))
		params->igpu = item->valueint;

	if (item = cJSON_GetObjectItem(object, "OUT"))
		strcpy(params->OUT, item->valuestring);
	if (item = cJSON_GetObjectItem(object, "out")) // accepted alias
		strcpy(params->OUT, item->valuestring);

	if (item = cJSON_GetObjectItem(object, "TerrainDir"))
		strcpy(params->TerrainDir, item->valuestring);

	if (item = cJSON_GetObjectItem(object, "SRTM90"))
		params->SRTM90 = item->valueint;
	if (item = cJSON_GetObjectItem(object, "srtm90")) // accepted alias
		params->SRTM90 = item->valueint;

	if (item = cJSON_GetObjectItem(object, "lonStart"))
		params->lonStart = item->valueint;
	if (item = cJSON_GetObjectItem(object, "latStart"))
		params->latStart = item->valueint;

	if (item = cJSON_GetObjectItem(object, "blockX"))
		params->blockX = item->valueint;
	if (item = cJSON_GetObjectItem(object, "blockY"))
		params->blockY = item->valueint;

	if (item = cJSON_GetObjectItem(object, "Depth(km)"))
		params->Depth = item->valuedouble;

	if (item = cJSON_GetObjectItem(object, "MLonStart"))
		params->MLonStart = item->valuedouble;
	if (item = cJSON_GetObjectItem(object, "MLatStart"))
		params->MLatStart = item->valuedouble;
	if (item = cJSON_GetObjectItem(object, "MLonEnd"))
		params->MLonEnd = item->valuedouble;
	if (item = cJSON_GetObjectItem(object, "MLatEnd"))
		params->MLatEnd = item->valuedouble;
	if (item = cJSON_GetObjectItem(object, "MLonStep"))
		params->MLonStep = item->valuedouble;
	if (item = cJSON_GetObjectItem(object, "MLatStep"))
		params->MLatStep = item->valuedouble;

	if (item = cJSON_GetObjectItem(object, "CrustLonStep"))
		params->CrustLonStep = item->valuedouble;
	if (item = cJSON_GetObjectItem(object, "CrustLatStep"))
		params->CrustLatStep = item->valuedouble;

	if (item = cJSON_GetObjectItem(object, "MVeticalStep"))
		params->MVeticalStep = item->valuedouble;

	if (item = cJSON_GetObjectItem(object, "MediumDir"))
		strcpy(params->MediumDir, item->valuestring);
	if (item = cJSON_GetObjectItem(object, "CrustDir"))
		strcpy(params->crustDir, item->valuestring);

	if (item = cJSON_GetObjectItem(object, "sourceX"))
		params->sourceX = item->valueint;
	if (item = cJSON_GetObjectItem(object, "sourceY"))
		params->sourceY = item->valueint;
	if (item = cJSON_GetObjectItem(object, "sourceZ"))
		params->sourceZ = item->valueint;

	if (item = cJSON_GetObjectItem(object, "sourceFile"))
		strcpy(params->sourceFile, item->valuestring);
	if (item = cJSON_GetObjectItem(object, "sourceDir"))
		strcpy(params->sourceDir, item->valuestring);
	if (item = cJSON_GetObjectItem(object, "degree2radian"))
		params->degree2radian = item->valueint;

	if (!params->useTerrain)
	{
		params->Depth = params->DH * params->NZ * 1e-3;
	}

	cJSON_Delete(object);
	free(jsonStr);

	return;
}
