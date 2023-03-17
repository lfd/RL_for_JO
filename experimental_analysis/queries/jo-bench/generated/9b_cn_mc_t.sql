SELECT * FROM company_name AS cn, movie_companies AS mc, title AS t WHERE cn.country_code = '[us]' AND mc.note LIKE '%(200%)%' AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND t.production_year BETWEEN 2007 AND 2010 AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_id = cn.id AND cn.id = mc.company_id;