SELECT * FROM role_type AS rt, movie_companies AS mc, cast_info AS ci, company_name AS cn WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND mc.note IS NOT NULL AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND rt.role = 'actress' AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.role_id = rt.id AND rt.id = ci.role_id;