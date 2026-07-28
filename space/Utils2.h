#pragma once


struct Vec2
{
	int x = 0, y = 0;
};

struct Vec2f
{
	float x = 0.0f, y = 0.0f;
};

struct Vec3
{
	int x = 0, y = 0, z = 0;
};

struct Vec3f
{
	float x = 0.f, y = 0.f, z = 0.f;
};

struct Vec4
{
	int x = 0, y = 0, z = 0, w = 0;
};

struct Vec4f
{
	float x = 0.f, y = 0.f, z = 0.f, w = 0.f;
};


struct Mat4f
{
	float m[4][4] = { { 0.f } };
	Mat4f()
	{
		for (int i = 0; i < 4; i++)
		{
			m[i][i] = 1.f;
		}
	}
};