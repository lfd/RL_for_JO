SELECT * FROM company_name AS cn, movie_companies AS mc, role_type AS rt, cast_info AS ci WHERE ci.note = '(voice: English version)' AND cn.country_code = '[jp]' AND mc.note LIKE '%(Japan)%' AND mc.note NOT LIKE '%(USA)%' AND (mc.note LIKE '%(2006)%' OR mc.note LIKE '%(2007)%') AND rt.role = 'actress' AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.role_id = rt.id AND rt.id = ci.role_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;