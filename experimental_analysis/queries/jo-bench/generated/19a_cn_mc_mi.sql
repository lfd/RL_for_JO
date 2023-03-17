SELECT * FROM company_name AS cn, movie_companies AS mc, movie_info AS mi WHERE cn.country_code = '[us]' AND mc.note IS NOT NULL AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;